"""
core.ingest_step2_indexing
==========================
Stage 2 of the ingestion pipeline:
1. Load staged chunks from 'data/staged_chunks/'.
2. Vectorize using Azure OpenAI (text-embedding-3-small).
3. Batch-upload to Azure AI Search.
"""

import os
import json
import hashlib
import time
import config.settings as settings
from services.azure_openai import get_embedding_client
from services.azure_search import get_search_client
from core.ingestion import create_index_if_not_exists

STAGED_DIR = "data/staged_chunks"

def push_staged_chunks_to_index():
    create_index_if_not_exists()
    embedding_client = get_embedding_client()
    search_client = get_search_client()

    if not os.path.exists(STAGED_DIR):
        print(f"❌ Staging directory {STAGED_DIR} not found.")
        return

    folders = [f for f in os.listdir(STAGED_DIR) if os.path.isdir(os.path.join(STAGED_DIR, f))]
    print(f"📂 Found {len(folders)} staged document folders.")

    for folder in folders:
        chunks_path = os.path.join(STAGED_DIR, folder, "chunks.json")
        info_path = os.path.join(STAGED_DIR, folder, "source_info.json")

        if not os.path.exists(chunks_path):
            print(f"⚠️  No chunks.json in {folder}, skipping.")
            continue

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        source_file = folder
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                source_file = json.load(f).get("source_file", folder)

        print(f"🚀 Indexing {len(chunks)} chunks for '{source_file}'...")

        batch = []
        BATCH_SIZE = 50
        MAX_CHARS = 10_000

        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            if not content: continue

            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS]

            # Vectorize
            print(f"  📊 Embedding chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
            emb = None
            for attempt in range(3):
                try:
                    emb = embedding_client.embeddings.create(
                        model=settings.EMBEDDING_MODEL,
                        inputs=[content],
                    ).data[0].embedding
                    print("✓")
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"\n  ⏳ Retrying chunk {i} (attempt {attempt+2}/3): {e}")
                        time.sleep(3)
                    else:
                        print(f"\n  ❌ Failed chunk {i} after 3 attempts: {e}")

            if emb is None: continue
            
            # Respect rate limits
            time.sleep(0.5) 

            chunk_id = hashlib.md5(f"{source_file}-{i}".encode()).hexdigest()

            batch.append({
                "id":             chunk_id,
                "content":        content,
                "parent_section": chunk.get("parent_section", "General"),
                "source_page":    chunk.get("source_page", 0),
                "source_file":    source_file,
                "chunk_type":     chunk.get("chunk_type", "text"),
                "vector":         emb,
            })

            if len(batch) >= BATCH_SIZE:
                print(f"  📤 Uploading batch...")
                search_client.upload_documents(batch)
                batch = []
        
        if batch:
            search_client.upload_documents(batch)

        print(f"✅ Finished indexing '{source_file}'")

if __name__ == "__main__":
    push_staged_chunks_to_index()
