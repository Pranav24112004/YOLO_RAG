# YOLO RAG Agent

A **Retrieval-Augmented Generation (RAG)** research assistant that answers questions about the YOLO (You Only Look Once) object-detection paper series. It ingests PDF research papers, chunks and vectorizes them, and uses a LangGraph agentic workflow to retrieve relevant context before generating grounded answers.

---

## Architecture Overview

```
User ─► Streamlit UI ─► FastAPI ─► LangGraph Agent ─► Azure AI Search (hybrid retrieval)
                                        │                       ▲
                                        ▼                       │
                                   Azure OpenAI            Ingestion Pipeline
                                  (GPT-4o-mini)           (Doc Intelligence +
                                                        text-embedding-3-small)
```

### Agent Workflow (LangGraph)

```
classify_intent
    ├─ greeting  → save_memory → END
    ├─ off_topic → save_memory → END
    └─ retrieve  → grade ─┬─ generate → save_memory → END
                          └─ retrieve  (retry, max 3)
```

1. **Intent Classification** — Routes the query to greeting, off-topic, or RAG pipeline.
2. **Retrieval** — Hybrid search (vector + semantic) on Azure AI Search.
3. **Grading** — LLM judges document relevance; retries retrieval up to 3× if documents are rejected.
4. **Generation** — Produces a cited answer from graded context.
5. **Memory** — Persists the conversation turn to Azure Cosmos DB.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Azure OpenAI — `o4-mini` (GPT-4o-mini) |
| **Embeddings** | Azure OpenAI — `text-embedding-3-small` (1 536 dims) |
| **Vision / Image chunking** | Azure OpenAI — `gpt-4.1-mini` |
| **Vector Store** | Azure AI Search (HNSW + semantic reranking) |
| **Document Parsing** | Azure Document Intelligence (`prebuilt-layout`) |
| **Blob Storage** | Azure Blob Storage |
| **Session Memory** | Azure Cosmos DB |
| **Agent Framework** | LangGraph + LangChain |
| **API** | FastAPI |
| **Frontend** | Streamlit |
| **Evaluation** | RAGAS (faithfulness, relevancy, precision, recall) |

---

## Project Structure

```
yolo_p2/
├── .env.example              # Required environment variables (template)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── config/
│   └── settings.py           # Centralized env-var loading & constants
│
├── services/                 # External service connectors
│   ├── azure_blob.py         # BlobServiceClient + SAS URL generation
│   ├── azure_search.py       # SearchClient & SearchIndexClient
│   ├── azure_cosmos.py       # CosmosClient (session history)
│   ├── azure_doc_intel.py    # DocumentAnalysisClient
│   └── azure_openai.py       # Embedding client + AzureChatOpenAI (LLM)
│
├── core/                     # Core business logic
│   ├── ingestion.py          # End-to-end: PDF → Doc Intel → chunk → vectorize → index
│   ├── ingest_step1_chunking.py   # Stage 1: chunk & save locally for review
│   ├── ingest_step2_indexing.py   # Stage 2: embed staged chunks & upload to search
│   ├── logger.py             # Logging utilities
│   └── chunking/             # Modular chunking pipeline
│       ├── pipeline.py       # Orchestrates text / table / image chunkers
│       ├── text_chunker.py   # Section-aware text chunking with overlap
│       ├── table_chunker.py  # Markdown table chunking (small vs. large tables)
│       └── image_chunker.py  # GPT-4.1 vision-based image description chunks
│
├── agent/                    # LangGraph agent
│   ├── state.py              # AgentState (TypedDict)
│   ├── prompts.py            # All LLM prompt templates
│   ├── graph.py              # StateGraph wiring & compilation
│   └── nodes/                # Individual graph nodes
│       ├── intent.py         # classify_intent, greeting, off_topic
│       ├── retriever.py      # Hybrid search retrieval
│       ├── grader.py         # Document relevance grading
│       ├── generator.py      # Answer generation with citations
│       └── memory.py         # Cosmos DB conversation save
│
├── api/                      # FastAPI backend
│   ├── main.py               # App instance, CORS, router registration
│   ├── routers/
│   │   └── chat.py           # POST /chat, GET /chat/{session_id}, GET /sessions, GET /source-link
│   └── schemas/
│       └── chat.py           # ChatRequest & ChatResponse Pydantic models
│
├── app.py                    # Streamlit frontend (chat UI + session sidebar)
├── run_chat.py               # CLI chat entry point (for local testing)
├── run_ingest.py             # Ingestion entry point (stage / index / all)
└── ragas_eval.py             # RAGAS evaluation script
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- An Azure subscription with the following services provisioned:
  - Azure AI Search
  - Azure OpenAI (with `text-embedding-3-small`, `o4-mini`, and `gpt-4.1-mini` deployments)
  - Azure Document Intelligence
  - Azure Blob Storage
  - Azure Cosmos DB
  - *(Optional)* Azure Application Insights

### 1. Clone & install

```bash
git clone <repo-url>
cd yolo_p2
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in all values in .env with your Azure credentials
```

Refer to [`.env.example`](.env.example) for the full list of required variables.

### 3. Upload PDFs

Upload your YOLO research-paper PDFs to the `papers` container in your Azure Blob Storage account.

### 4. Run ingestion

The ingestion pipeline has two stages that can be run independently or together:

```bash
# Stage 1 — Parse PDFs & chunk (saves to data/staged_chunks/ for review)
python run_ingest.py stage

# Stage 2 — Embed staged chunks & upload to Azure AI Search
python run_ingest.py index

# Or run both stages in sequence
python run_ingest.py all
```

#### Ingestion Assumptions

- PDFs are stored in the `papers` Blob container.
- Document Intelligence results are cached in a `doc-intelligence-save` Blob container (auto-created).
- Page images for vision chunking are stored in the `papers-page-i` container.
- Chunks exceeding 10 000 characters are truncated before embedding.
- Embedding calls include rate-limit retry logic (3 attempts with 3 s backoff).

### 5. Start the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 6. Start the Streamlit frontend

```bash
streamlit run app.py
```

The UI connects to the FastAPI backend at `http://localhost:8000/api/v1`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/chat` | Send a message and receive an agent response |
| `GET` | `/api/v1/chat/{session_id}` | Retrieve chat history for a session |
| `GET` | `/api/v1/sessions` | List all stored sessions |
| `GET` | `/api/v1/source-link?file=...&page=...` | Generate a time-limited SAS link to a source PDF |

---

## Evaluation

Run the RAGAS evaluation suite against a golden dataset of YOLO questions:

```bash
python ragas_eval.py
```

Metrics computed: **Faithfulness**, **Answer Relevancy**, **Context Precision**, **Context Recall**.

---

## Environment Variables

| Variable | Description |
|---|---|
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search endpoint URL |
| `AZURE_SEARCH_KEY` | Azure AI Search admin key |
| `AZURE_DOC_INTEL_ENDPOINT` | Azure Document Intelligence endpoint |
| `AZURE_DOC_INTEL_KEY` | Document Intelligence API key |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage connection string |
| `AZURE_COSMOS_ENDPOINT` | Cosmos DB endpoint |
| `AZURE_COSMOS_KEY` | Cosmos DB primary key |
| `AZURE_OPENAI_4.1_ENDPOINT` | Azure OpenAI GPT-4.1-mini deployment endpoint (full URL) |
| `AZURE_OPENAI_4.1_KEY` | API key for GPT-4.1-mini deployment |
| `AZURE_TEXT_EMBEDDING_ENDPOINT` | Azure OpenAI text-embedding-3-small deployment endpoint (full URL) |
| `AZURE_TEXT_EMBEDDING_KEY` | API key for text-embedding-3-small deployment |
| `AZURE_GPT_4O_MINI_ENDPOINT` | Azure OpenAI o4-mini deployment endpoint (full URL) |
| `AZURE_GPT_4O_MINI_KEY` | API key for o4-mini deployment |
| `AZURE_ANALYTICS_CONNECTION_STRING` | *(Optional)* Application Insights connection string |

---

## License

This project is for educational and research purposes.
