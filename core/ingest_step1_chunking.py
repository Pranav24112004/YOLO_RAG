"""
core.ingest_step1_chunking
==========================
Stage 1 of the ingestion pipeline:
1. Fetch PDFs from Blob Storage.
2. Run Doc Intelligence (with caching).
3. Chunk using specialized chunkers.
4. Save chunks locally for manual review/editing.
"""

import os
import json
from services.azure_blob import get_blob_service_client
from services.azure_doc_intel import get_doc_intel_client
from core.chunking.pipeline import run_pipeline
from core.ingestion import get_or_analyze_document, _get_oai_client

STAGED_DIR = "data/staged_chunks"

def stage_chunks_for_all_files():
    blob_service = get_blob_service_client()
    doc_client = get_doc_intel_client()
    oai_client = _get_oai_client()
    container = blob_service.get_container_client("papers")

    # Ensure staging directory exists
    os.makedirs(STAGED_DIR, exist_ok=True)

    blobs = list(container.list_blobs())
    print(f"🔍 Found {len(blobs)} blobs in container 'papers'.")

    for blob in blobs:
        blob_name = blob.name
        safe_name = blob_name.replace(".pdf", "").replace(" ", "_")
        target_dir = os.path.join(STAGED_DIR, safe_name)
        os.makedirs(target_dir, exist_ok=True)
        output_path = os.path.join(target_dir, "chunks.json")

        print(f"📖 Processing {blob_name}...")
        
        # 1. Download PDF
        pdf_bytes = container.get_blob_client(blob_name).download_blob().readall()

        # 2. Doc Intel
        raw_json = get_or_analyze_document(blob_name, pdf_bytes, doc_client, blob_service)

        # 3. Chunking
        paper_title = blob_name.replace(".pdf", "").replace("_", " ")
        chunks = run_pipeline(
            pdf_bytes=pdf_bytes,
            raw_json=raw_json,
            oai_client=oai_client,
            paper_title=paper_title,
            output_path=None,
            print_progress=True
        )

        # 4. Save locally
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        # Also save a 'source_info.json' for metadata
        with open(os.path.join(target_dir, "source_info.json"), "w", encoding="utf-8") as f:
            json.dump({"source_file": blob_name, "chunk_count": len(chunks)}, f, indent=2)

        print(f"✅ Staged {len(chunks)} chunks at: {output_path}")

if __name__ == "__main__":
    stage_chunks_for_all_files()
