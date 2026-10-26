import sys
from core.ingest_step1_chunking import stage_chunks_for_all_files
from core.ingest_step2_indexing import push_staged_chunks_to_index

def main():
    """
    Entry point for the two-stage ingestion pipeline.
    
    Modes:
      stage: Run Doc Intelligence and chunking, save results to data/staged_chunks/ (no embeddings).
      index: Read JSON from data/staged_chunks/, generate embeddings, and upload to Azure AI Search.
      all:   Run both stages sequentially.
    """
    if len(sys.argv) < 2:
        print("Usage: python run_ingest.py [stage|index|all]")
        return

    mode = sys.argv[1].lower()

    if mode == "stage":
        print("  STAGE 1: Extracted chunks (PDF -> JSON)...")
        stage_chunks_for_all_files()
    elif mode == "index":
        print("  STAGE 2: Indexing (JSON -> AI Search)...")
        push_staged_chunks_to_index()
    elif mode == "all":
        print("  FULL PIPELINE: Staging then Indexing...")
        stage_chunks_for_all_files()
        push_staged_chunks_to_index()
    else:
        print(f"Unknown mode: {mode}")
        print("Valid modes: stage, index, all")

if __name__ == "__main__":
    main()
