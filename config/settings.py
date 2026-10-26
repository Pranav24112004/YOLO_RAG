
# Configuration Module
# This file centralizes environment variable loading and constant definitions.

import os
from dotenv import load_dotenv

load_dotenv()

# --- AZURE SERVICES ---
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "yolo-papers-index"

AZURE_COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
AZURE_COSMOS_KEY = os.getenv("AZURE_COSMOS_KEY")
DATABASE_NAME = "yolo-sessions"
CONTAINER_NAME = "sessions-db"

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
AZURE_DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTEL_KEY") 

# --- AZURE OPENAI (Embeddings) ---
AZURE_TEXT_EMBEDDING_ENDPOINT = os.getenv("AZURE_TEXT_EMBEDDING_ENDPOINT")
AZURE_TEXT_EMBEDDING_KEY      = os.getenv("AZURE_TEXT_EMBEDDING_KEY")
EMBEDDING_MODEL               = "text-embedding-3-small"

# --- AZURE OPENAI (GPT-4o-mini - Chat / LLM) ---
AZURE_GPT_4O_MINI_ENDPOINT = os.getenv("AZURE_GPT_4O_MINI_ENDPOINT")
AZURE_GPT_4O_MINI_KEY      = os.getenv("AZURE_GPT_4O_MINI_KEY")
LLM_DEPLOYMENT              = "o4-mini"
LLM_API_VERSION             = "2025-01-01-preview"

# --- SETTINGS ---
VECTOR_DIMENSIONS = 1536

# --- AZURE OPENAI (Vision / GPT-4.1) ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_4.1_ENDPOINT")
AZURE_OPENAI_KEY      = os.getenv("AZURE_OPENAI_4.1_KEY")
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"

# --- CHUNKING ---
BLOB_CONTAINER        = "papers"
PAGES_CONTAINER       = "papers-page-i"
TABLE_CELL_THRESHOLD  = 100     # <=100 cells -> full Markdown, >100 -> row-per-chunk
FLUSH_CHARS           = 1500    # ~250 words before flushing a text chunk
CHUNK_OVERLAP         = 300     # ~50 words to carry over between chunks
MIN_IMAGE_AREA_PX     = 5000    # skip tiny decorative images (width_pt * height_pt)
