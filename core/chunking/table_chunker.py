"""
core.chunking.table_chunker
============================
Converts a Doc Intelligence table dict into chunk(s).

Strategy
--------
*  <= TABLE_CELL_THRESHOLD cells  →  one chunk with the full Markdown table
*  >  TABLE_CELL_THRESHOLD cells  →  one chunk per data row, each prefixed with
   a sticky header row so the LLM always has column context.

Why 100 cells?
  Worst-case small (10×10 = 100):  ~1,500 chars ≈ 375 tokens  → safe
  Table 4 (21×22 = 462 cells):     ~3,567 chars ≈ 891 tokens  → LLM loses
  column→value mapping across 22 wide columns; row-per-chunk fixes this.
"""

from config.settings import TABLE_CELL_THRESHOLD


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_row_map(tbl: dict) -> dict[int, list]:
    rows: dict[int, list] = {}
    for cell in tbl["cells"]:
        rows.setdefault(cell["row_index"], []).append(cell)
    return rows


def _cell_text(cell: dict) -> str:
    # Clean selection marks and newlines
    text = cell["content"].replace("\n", " ")
    text = text.replace(":selected:", "").replace(":unselected:", "")
    return text.strip()


def _get_table_matrix(tbl: dict) -> list[list[str]]:
    """Build a full grid including spans."""
    rows = tbl.get("row_count", 1)
    cols = tbl.get("column_count", 1)
    matrix = [["" for _ in range(cols)] for _ in range(rows)]
    
    for cell in tbl["cells"]:
        r, c = cell["row_index"], cell["column_index"]
        rs, cs = cell.get("row_span", 1), cell.get("column_span", 1)
        txt = _cell_text(cell)
        
        for ri in range(r, r + rs):
            for ci in range(c, c + cs):
                if ri < rows and ci < cols:
                    matrix[ri][ci] = txt
    return matrix


def render_table_full(tbl: dict) -> str:
    """Render entire table as a GitHub-flavoured Markdown table string."""
    matrix = _get_table_matrix(tbl)
    lines: list[str] = []

    for row_idx, row_parts in enumerate(matrix):
        lines.append("| " + " | ".join(row_parts) + " |")
        if row_idx == 0:                            # separator after header
            col_count = len(row_parts)
            lines.append("|" + "|".join(["---"] * col_count) + "|")

    return "\n".join(lines)


def iter_table_row_chunks(tbl: dict, section: str, page: int):
    """
    Yield one chunk dict per data row (row_index > 0), each including the
    sticky header + separator so the LLM has column context in isolation.
    """
    matrix = _get_table_matrix(tbl)
    if not matrix: return
    
    header_row = matrix[0]
    col_count  = len(header_row)
    header_md  = "| " + " | ".join(header_row) + " |"
    sep_md     = "|" + "|".join(["---"] * col_count) + "|"

    for row_idx in range(1, len(matrix)):
        data_md = "| " + " | ".join(matrix[row_idx]) + " |"
        yield {
            "chunk_type":     "table_row",
            "parent_section": section,
            "source_page":    page,
            "y_order":        0.0,
            "content": (
                f"[Section: {section}]\n"
                f"{header_md}\n{sep_md}\n{data_md}"
            ),
        }


# ── Public API ────────────────────────────────────────────────────────────────

def process_table(tbl: dict, section: str, page: int) -> list[dict]:
    """
    Convert one Doc Intelligence table dict into one or more chunk dicts.
    Decision is based on TABLE_CELL_THRESHOLD from settings.
    """
    n_cells = len(tbl["cells"])

    if n_cells <= TABLE_CELL_THRESHOLD:
        return [{
            "chunk_type":     "table",
            "parent_section": section,
            "source_page":    page,
            "y_order":        0.0,
            "content":        f"[Section: {section}]\n[TABLE]\n{render_table_full(tbl)}",
        }]
    else:
        return list(iter_table_row_chunks(tbl, section, page))
