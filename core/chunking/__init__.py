"""
core.chunking
=============
Modular chunking pipeline for research-paper PDFs processed through
Azure Document Intelligence.

Sub-modules
-----------
text_chunker      — paragraphs → text chunks (flushed at FLUSH_CHARS)
table_chunker     — tables → full Markdown (small) or row-per-chunk (large)
image_chunker     — PDF vector figures → GPT-vision descriptions
citation_chunker  — References section → individual citation chunks
pipeline          — orchestrates all four chunkers into one sorted output
"""
