"""
core.ingestion
==============
End-to-end ingestion pipeline:
  1. Create / verify Azure AI Search index
  2. For each PDF in Blob Storage:
     a. Run Doc Intelligence (with blob-based caching)
     b. Chunk with core.chunking.pipeline (text + table + image)
     c. Vectorize chunks with Azure OpenAI embeddings
     d. Batch-upload to the search index
"""

import hashlib
import json
import time
from typing import List

import config.settings as settings
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)
from openai import AzureOpenAI
from services.azure_blob import get_blob_service_client
from services.azure_doc_intel import get_doc_intel_client
from services.azure_search import get_index_client, get_search_client
from services.azure_openai import get_embedding_client
from core.chunking.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# INDEX MANAGEMENT
# ---------------------------------------------------------------------------

def create_index_if_not_exists():
    index_client = get_index_client()
    try:
        index_client.get_index(settings.INDEX_NAME)
        print("Index already exists.")
    except Exception:
        print("Creating index...")
        fields = [
            SimpleField(name="id",              type="Edm.String",  key=True),
            SearchableField(name="content",     type="Edm.String"),
            SearchableField(name="parent_section", type="Edm.String", filterable=True),
            SimpleField(name="source_page",     type="Edm.Int32", filterable=True),
            SimpleField(name="source_file",     type="Edm.String", filterable=True),
            SimpleField(name="chunk_type",      type="Edm.String",
                        filterable=True),
            SearchField(
                name="vector",
                type="Collection(Edm.Single)",
                searchable=True,
                vector_search_dimensions=settings.VECTOR_DIMENSIONS,
                vector_search_profile_name="my-vector-profile",
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(
                name="my-vector-profile",
                algorithm_configuration_name="hnsw",
            )],
        )
        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="my-semantic-config",
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[SemanticField(field_name="content")],
                        keywords_fields=[SemanticField(field_name="parent_section")],
                    ),
                )
            ]
        )
        index = SearchIndex(
            name=settings.INDEX_NAME,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
        index_client.create_index(index)
        print("Index created.")


# ---------------------------------------------------------------------------
# DOC INTELLIGENCE CACHE  (stores full to_dict() output)
# ---------------------------------------------------------------------------

def get_or_analyze_document(
    blob_name: str,
    pdf_bytes: bytes,
    doc_client,
    blob_service_client,
) -> dict:
    """
    Checks 'doc-intelligence-save' container for a cached JSON result.
    Cache HIT  → loads and returns the full to_dict() JSON.
    Cache MISS → runs Doc Intelligence, saves to_dict(), returns dict.
    """
    cache_container_name = "doc-intelligence-save"
    cache_blob_name = blob_name.replace(".pdf", ".json").replace(" ", "_")

    # Ensure cache container exists
    try:
        cache_container = blob_service_client.get_container_client(cache_container_name)
        cache_container.get_container_properties()
    except Exception:
        print(f"  📦 Creating cache container '{cache_container_name}'...")
        cache_container = blob_service_client.create_container(cache_container_name)

    # Try loading from cache
    cache_blob = cache_container.get_blob_client(cache_blob_name)
    try:
        cached_bytes = cache_blob.download_blob().readall()
        print(f"  ✅ Cache HIT for '{blob_name}'. Loading from '{cache_blob_name}'...")
        return json.loads(cached_bytes.decode("utf-8"))
    except Exception:
        print(f"  🔍 Cache MISS for '{blob_name}'. Running Document Intelligence...")

    # Run Doc Intelligence
    poller = doc_client.begin_analyze_document("prebuilt-layout", pdf_bytes)
    result = poller.result()

    # Use the full SDK to_dict() — preserves paragraphs, tables, figures, etc.
    serialized = result.to_dict()

    # Save to cache
    cache_blob.upload_blob(
        json.dumps(serialized, ensure_ascii=False, indent=2).encode("utf-8"),
        overwrite=True,
    )
    print(f"  💾 Saved Doc Intelligence result to cache as '{cache_blob_name}'.")
    return serialized


# ---------------------------------------------------------------------------
# AZURE OPENAI CLIENT  (for image chunking / GPT-vision)
# ---------------------------------------------------------------------------

def _get_oai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )


# ---------------------------------------------------------------------------
# MAIN FILE PROCESSOR
# ---------------------------------------------------------------------------

def process_file(blob_name: str, container_client, blob_service_client) -> None:
    doc_client    = get_doc_intel_client()
    embedding_client = get_embedding_client()
    search_client = get_search_client()
    oai_client    = _get_oai_client()

    # ── 1. Download PDF ───────────────────────────────────────────────────
    print(f"📖 Reading {blob_name}...")
    pdf_bytes = container_client.get_blob_client(blob_name).download_blob().readall()

    # ── 2. Get Doc Intelligence result (cached or fresh) ──────────────────
    raw_json = get_or_analyze_document(
        blob_name, pdf_bytes, doc_client, blob_service_client,
    )

    # ── 3. Run the new chunking pipeline (text + table + image) ───────────
    paper_title = blob_name.replace(".pdf", "").replace("_", " ")
    print(f"  🧠 Running chunking pipeline for '{blob_name}'...")

    chunks = run_pipeline(
        pdf_bytes=pdf_bytes,
        raw_json=raw_json,
        oai_client=oai_client,
        paper_title=paper_title,
        output_path=None,       # don't write intermediate JSON
        print_progress=True,
    )
    print(f"  📄 {len(chunks)} chunks produced.")

    # ── 4. Vectorize & batch-upload to Azure AI Search ────────────────────
    safe_base  = blob_name.replace(".pdf", "").replace(".", "-").replace(" ", "_")
    batch: list = []
    BATCH_SIZE  = 50
    MAX_CHARS   = 10_000   # safety truncation before embedding

    for i, chunk in enumerate(chunks):
        content = chunk["content"]

        # Safety truncation
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS]
            print(f"  ⚠️  Chunk {i} truncated to {MAX_CHARS} chars.")

        # Vectorize with retry on rate-limit
        emb = None
        for attempt in range(3):
            try:
                emb = embedding_client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    inputs=[content],
                ).data[0].embedding
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  ⏳ Rate limit / error on chunk {i}, retrying in 3s... ({e})")
                    time.sleep(3)
                else:
                    print(f"  ❌ Skipping chunk {i} after 3 failures: {e}")

        if emb is None:
            continue

        time.sleep(0.5)

        chunk_id = hashlib.md5(f"{blob_name}-{i}".encode()).hexdigest()

        batch.append({
            "id":             chunk_id,
            "content":        content,
            "parent_section": chunk.get("parent_section", "General"),
            "source_page":    chunk.get("source_page", 0),
            "source_file":    blob_name,
            "chunk_type":     chunk.get("chunk_type", "text"),
            "vector":         emb,
        })

        if len(batch) >= BATCH_SIZE:
            print(f"  🚀 Uploading batch of {len(batch)} chunks...")
            search_client.upload_documents(batch)
            batch = []
            time.sleep(0.5)

    # Final flush
    if batch:
        print(f"  🚀 Uploading final batch of {len(batch)} chunks...")
        search_client.upload_documents(batch)

    print(f"  ✅ Done: {blob_name}")


# ---------------------------------------------------------------------------
# PIPELINE ENTRY POINT
# ---------------------------------------------------------------------------

def run_ingestion_pipeline() -> None:
    create_index_if_not_exists()
    blob_service = get_blob_service_client()
    container    = blob_service.get_container_client("papers")

    blobs = list(container.list_blobs())
    print(f"🔍 Found {len(blobs)} blobs in container 'papers':")
    for b in blobs:
        print(f"   - {b.name}")

    for blob in blobs:
        process_file(blob.name, container, blob_service)


if __name__ == "__main__":
    run_ingestion_pipeline()
