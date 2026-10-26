"""
core.chunking.text_chunker
===========================
Converts Doc Intelligence paragraph + table items into text / table chunk dicts.

Algorithm
---------
1. Build set of character-offset ranges that belong to table spans
   (so table-interior paragraphs are skipped).
2. Merge paragraphs + table items into a single offset-sorted stream.
3. Walk the stream:
   - Heading paragraphs (title / sectionHeading)  → flush buffer, update section.
   - Body paragraphs                               → accumulate in text_buffer.
   - Buffer >= FLUSH_CHARS                         → flush as text chunk.
   - Table items                                   → flush buffer, delegate to
                                                     table_chunker.process_table.
4. Track `section_by_page` mapping for downstream use by image_chunker.

Returns
-------
(chunks, section_by_page)
  chunks          : list[dict]  — text + table + table_row chunks
  section_by_page : dict[int, str]  — page_num → last section heading seen
"""

from config.settings import FLUSH_CHARS, CHUNK_OVERLAP
from .table_chunker import process_table

SKIP_ROLES    = {"pageNumber", "pageHeader"}
HEADING_ROLES = {"title", "sectionHeading"}

# ── Offset helpers ─────────────────────────────────────────────────────────────

def build_table_offset_ranges(tables: list[dict]) -> list[tuple[int, int]]:
    ranges = []
    for tbl in tables:
        for span in tbl["spans"]:
            s = span["offset"]
            ranges.append((s, s + span["length"]))
    return ranges


def is_inside_table(offset: int, length: int, ranges: list[tuple[int, int]]) -> bool:
    p_s, p_e = offset, offset + length
    return any(p_s < t_e and p_e > t_s for (t_s, t_e) in ranges)


# ── Main function ──────────────────────────────────────────────────────────────

def extract_text_and_table_chunks(
    paragraphs: list[dict],
    tables: list[dict],
) -> tuple[list[dict], dict[int, str]]:
    """
    Produce text/table/table_row chunks and a page→section mapping.

    Parameters
    ----------
    paragraphs : raw 'paragraphs' array from Doc Intelligence JSON
    tables     : raw 'tables' array from Doc Intelligence JSON

    Returns
    -------
    chunks          : list[dict]
    section_by_page : dict[int, str]
    """
    table_offset_ranges = build_table_offset_ranges(tables)

    # ── Build offset-sorted item stream ───────────────────────────────────
    stream: list[dict] = []

    for para in paragraphs:
        role = para.get("role")
        if role in SKIP_ROLES:
            continue
        span = para["spans"][0] if para["spans"] else {"offset": 0, "length": 0}
        if is_inside_table(span["offset"], span["length"], table_offset_ranges):
            continue
        pg = para["bounding_regions"][0]["page_number"] if para["bounding_regions"] else 0
        stream.append({
            "_offset": span["offset"], "_page": pg,
            "_is_table": False, "role": role, "content": para["content"],
        })

    for tbl in tables:
        first_off = tbl["spans"][0]["offset"] if tbl["spans"] else 0
        pg        = tbl["bounding_regions"][0]["page_number"] if tbl["bounding_regions"] else 0
        stream.append({"_offset": first_off, "_page": pg, "_is_table": True, "table": tbl})

    stream.sort(key=lambda x: x["_offset"])

    # ── Walk stream ───────────────────────────────────────────────────────
    chunks: list[dict]       = []
    section_by_page: dict    = {}
    current_section          = "Title Page"
    text_buffer              = ""
    buffer_page              = 1
    
    def flush_text(keep_overlap: bool = True) -> None:
        nonlocal text_buffer
        stripped = text_buffer.strip()
        # Drop junk or very small clusters
        if len(stripped) >= 40:
            chunks.append({
                "chunk_type":     "text",
                "parent_section": current_section,
                "source_page":    buffer_page,
                "y_order":        0.0,
                "content":        f"[Section: {current_section}]\n{stripped}",
            })
        
        if keep_overlap and len(stripped) > CHUNK_OVERLAP:
            # Look for a space or newline within the overlap window to avoid splitting words
            # But for simplicity, we'll just take the last N chars for now
            text_buffer = stripped[-CHUNK_OVERLAP:]
        else:
            text_buffer = ""

    for item in stream:
        pg = item["_page"]
        
        # Ensure we always have a section for the page
        if pg not in section_by_page:
            section_by_page[pg] = current_section

        if item["_is_table"]:
            flush_text()
            for chunk in process_table(item["table"], current_section, pg):
                chunks.append(chunk)
            continue

        role    = item.get("role")
        content = item["content"]

        if role in HEADING_ROLES:
            flush_text(keep_overlap=False)  # Clear completely for new section
            current_section    = content
            buffer_page        = pg
            section_by_page[pg] = current_section
            continue

        # If we see a new page, maybe flush if the buffer is getting old
        if pg != buffer_page and len(text_buffer) > 500:
            flush_text()
            buffer_page = pg

        text_buffer += " " + content
        if len(text_buffer) >= FLUSH_CHARS:
            flush_text()

    flush_text(keep_overlap=False)
    return chunks, section_by_page
