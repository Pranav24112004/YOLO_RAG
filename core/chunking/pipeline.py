"""
core.chunking.pipeline
=======================
Orchestrates all four chunkers into a single, document-order output list.

Chunk types in the output
--------------------------
  "text"       — body text flushed at FLUSH_CHARS
  "table"      — small table (<=TABLE_CELL_THRESHOLD cells) as full Markdown
  "table_row"  — one data row of a large table with sticky header
  "image"      — GPT-described figure with spatial before/after context
  "citation"   — individual reference from the References section

Sort order
----------
All chunks are sorted by (source_page, y_order).
  - text / table / table_row have y_order = 0.0  → appear at page top
  - image chunks have y_order = y_top_in (inches) → slot at actual position
  - citation chunks have y_order = citation_number → sorted at end

Public API
----------
  run_pipeline(pdf_bytes, raw_json, oai_client, paper_title, output_path)
      -> list[dict]   (also writes JSON to output_path if given)
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Optional

from openai import AzureOpenAI

from .text_chunker     import extract_text_and_table_chunks
from .image_chunker    import build_page_para_spatial, extract_image_chunks


def run_pipeline(
    pdf_bytes:    bytes,
    raw_json:     dict,
    oai_client:   AzureOpenAI,
    paper_title:  str = "Research Paper",
    output_path:  Optional[str] = None,
    print_progress: bool = True,
) -> list[dict]:
    """
    Full chunking pipeline.

    Parameters
    ----------
    pdf_bytes     : raw PDF bytes
    raw_json      : parsed Doc Intelligence JSON (result of json.load / to_dict)
    oai_client    : AzureOpenAI client
    paper_title   : paper title used in the vision system prompt
    output_path   : if provided, write final chunks JSON to this path
    print_progress: log progress to stdout

    Returns
    -------
    list[dict]  — all chunks sorted and indexed
    """
    paragraphs = raw_json["paragraphs"]
    tables     = raw_json["tables"]

    # ── 1. Text + Table chunks ─────────────────────────────────────────────
    if print_progress:
        print("[pipeline] Step 1/4 — text & table chunking...")
    text_table_chunks, section_by_page = extract_text_and_table_chunks(paragraphs, tables)
    if print_progress:
        print(f"           {len(text_table_chunks)} text/table chunks")

    # ── 2. Image chunks ────────────────────────────────────────────────────
    if print_progress:
        print("[pipeline] Step 2/4 — image extraction & GPT-vision...")
    page_para_spatial = build_page_para_spatial(paragraphs)
    image_chunks = extract_image_chunks(
        pdf_bytes, page_para_spatial, section_by_page,
        oai_client, paper_title, print_progress,
    )
    if print_progress:
        print(f"           {len(image_chunks)} image chunks")

    # ── 3. Merge + sort + index ────────────────────────────────────────────
    if print_progress:
        print("[pipeline] Step 3/3 — merging & sorting...")

    all_chunks = text_table_chunks + image_chunks
    all_chunks.sort(key=lambda c: (c["source_page"], c["y_order"]))

    for i, chunk in enumerate(all_chunks):
        chunk["chunk_index"] = i

    # ── 5. Summary ─────────────────────────────────────────────────────────
    counts = Counter(c["chunk_type"] for c in all_chunks)
    if print_progress:
        print(f"\n{'='*50}")
        print(f"TOTAL CHUNKS : {len(all_chunks)}")
        for ct, n in counts.items():
            print(f"  {ct:<12}: {n}")

    # ── 6. Save ────────────────────────────────────────────────────────────
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(all_chunks, fh, ensure_ascii=False, indent=2)
        if print_progress:
            print(f"Saved to     : {output_path}")

    return all_chunks
