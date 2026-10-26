"""
core.chunking.image_chunker
============================
Extracts figures from a PDF and generates GPT-vision descriptions.

Why get_drawings() instead of get_images()
------------------------------------------
The YOLOv1 paper (and most arXiv PDFs) render diagrams, charts, and system
figures as PDF vector paths — not as embedded bitmap images.  get_images()
only finds embedded PNGs/JPEGs (logos, photos); get_drawings() returns every
line/rect/curve path.

Approach
---------
1. For each page, collect bounding rects of all drawing paths.
2. Cluster rects that are within GAP points of each other (simple greedy merge).
3. Filter clusters smaller than MIN_IMAGE_AREA_PX.
4. For each remaining cluster, render the region at 2× resolution as PNG.
5. Call the Azure OpenAI vision model with the cropped image + spatial context.
6. Return a list of 'image' chunk dicts.

Spatial context
---------------
Rather than using the whole page as context, we cross-reference the image's
y-position (in inches) against Doc Intelligence paragraph bounding_regions to
find the 2 paragraphs immediately above and the 2 paragraphs immediately below
the figure on the same page.  This gives the LLM precise, minimal context.
"""

import base64
import sys
import time
from typing import Optional

import fitz

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config.settings import (
    MIN_IMAGE_AREA_PX,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
)

DRAWING_GAP_PT    = 15      # points — cluster nearby drawing paths together
RENDER_SCALE      = 2       # 2× resolution for rendered figure PNG
MAX_CONTEXT_CHARS = 250     # chars of surrounding paragraph text to include
VISION_MAX_TOKENS = 800


SKIP_ROLES = {"pageNumber", "pageHeader"}

SYSTEM_PROMPT_TMPL = (
    'Analyse figures from the paper "{title}".'
    " Respond in numbered lines only. No preamble."
    " Graph/plot: 1.type 2.axes 3.series 4.key finding."
    " Diagram: 1.type 2.components 3.labels."
    " Photo/result: 1.content 2.annotations 3.observation."
    " Unreadable/decorative: output only SKIP."
)


# ── Spatial index ──────────────────────────────────────────────────────────────

def build_page_para_spatial(paragraphs: list[dict]) -> dict[int, list]:
    """
    Returns {page_num: [(y_top, y_bottom, content), ...]} sorted by y_top.
    Used to find paragraphs above/below an image bounding box.
    """
    index: dict[int, list] = {}
    for para in paragraphs:
        if para.get("role") in SKIP_ROLES:
            continue
        if not para.get("bounding_regions"):
            continue
        br   = para["bounding_regions"][0]
        pg   = br["page_number"]
        poly = br["polygon"]
        y_t  = min(pt["y"] for pt in poly)
        y_b  = max(pt["y"] for pt in poly)
        index.setdefault(pg, []).append((y_t, y_b, para["content"]))
    for pg in index:
        index[pg].sort(key=lambda x: x[0])
    return index


def get_spatial_context(
    page: int,
    img_y_top: float,
    img_y_bottom: float,
    page_para_spatial: dict[int, list],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, str]:
    """
    Returns (before_text, after_text) using Doc Intel paragraph y-coords.
    'before' = nearest 2 paragraphs whose bottom edge is above the image top.
    'after'  = nearest 2 paragraphs whose top edge is below the image bottom.
    """
    paras  = page_para_spatial.get(page, [])
    before = [c for (yt, yb, c) in paras if yb <= img_y_top]
    after  = [c for (yt, yb, c) in paras if yt >= img_y_bottom]
    return " ".join(before[-2:])[:max_chars], " ".join(after[:2])[:max_chars]


# ── GPT vision call ────────────────────────────────────────────────────────────

def call_vision(
    page: int,
    img_num: int,
    b64_uri: str,
    before: str,
    after: str,
    oai_client,
    paper_title: str,
) -> str:
    """Call the Azure OpenAI vision endpoint and return the description string."""
    system_prompt = SYSTEM_PROMPT_TMPL.format(title=paper_title)
    user_text = (
        f"Figure on page {page} (image #{img_num}).\n"
        f"CONTEXT BEFORE: {before or '[none]'}\n"
        f"CONTEXT AFTER: {after or '[none]'}\n"
        "Describe this figure."
    )
    # Rate Limit Handling (Limit: 250 RPM)
    time.sleep(0.5) 

    try:
        resp = oai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text",      "text": user_text},
                    {"type": "image_url", "image_url": {"url": b64_uri, "detail": "low"}},
                ]},
            ],
            max_completion_tokens=VISION_MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"      [Error calling GPT-4.1: {e}]")
        time.sleep(5) # Backoff
        return "SKIP"


# ── Drawing-based figure extraction ───────────────────────────────────────────

def _cluster_drawing_rects(page_obj: fitz.Page) -> list[fitz.Rect]:
    """
    Return merged bounding-box clusters from all drawing paths on the page.
    Clusters within DRAWING_GAP_PT of each other are merged.
    """
    drawings  = page_obj.get_drawings()
    draw_rects = [
        d["rect"] for d in drawings
        if d["rect"].is_valid and not d["rect"].is_empty
    ]

    clusters: list[fitz.Rect] = []
    for r in draw_rects:
        merged = False
        for i, c in enumerate(clusters):
            expanded = fitz.Rect(
                c.x0 - DRAWING_GAP_PT, c.y0 - DRAWING_GAP_PT,
                c.x1 + DRAWING_GAP_PT, c.y1 + DRAWING_GAP_PT,
            )
            if expanded.intersects(r):
                clusters[i] = c | r   # union
                merged = True
                break
        if not merged:
            clusters.append(fitz.Rect(r))

    return [c for c in clusters if c.width * c.height >= MIN_IMAGE_AREA_PX]


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_image_chunks(
    pdf_bytes: bytes,
    page_para_spatial: dict[int, list],
    section_by_page: dict[int, str],
    oai_client,
    paper_title: str = "Research Paper",
    print_progress: bool = True,
) -> list[dict]:
    """
    Extract all figure regions from a PDF, describe them via GPT-vision,
    and return a list of 'image' chunk dicts.

    Parameters
    ----------
    pdf_bytes         : raw PDF bytes (downloaded from blob)
    page_para_spatial : output of build_page_para_spatial()
    section_by_page   : {page_num: section_heading} from text_chunker
    oai_client        : AzureOpenAI client instance
    paper_title       : used in the system prompt
    print_progress    : log each step to stdout
    """
    pdf   = fitz.open(stream=pdf_bytes, filetype="pdf")
    image_chunks: list[dict] = []

    for page_num in range(len(pdf)):
        page_obj  = pdf[page_num]
        page_1idx = page_num + 1
        clusters  = _cluster_drawing_rects(page_obj)

        if not clusters:
            continue

        if print_progress:
            print(f"  Page {page_1idx}: {len(clusters)} figure region(s)")

        # Best-effort section guess: look current page, then previous page
        page_section = section_by_page.get(
            page_1idx,
            section_by_page.get(page_1idx - 1, "Introduction"),
        )

        for img_idx, cluster_rect in enumerate(clusters):
            y_top_in    = cluster_rect.y0 / 72.0
            y_bottom_in = cluster_rect.y1 / 72.0

            # Render at 2× as PNG
            mat    = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
            pixmap = page_obj.get_pixmap(matrix=mat, clip=cluster_rect, alpha=False)
            img_bytes = pixmap.tobytes("png")
            b64_uri   = f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"

            before, after = get_spatial_context(page_1idx, y_top_in, y_bottom_in, page_para_spatial)

            if print_progress:
                print(
                    f"    region {img_idx+1}: {cluster_rect.width:.0f}x{cluster_rect.height:.0f}pt "
                    f"-> {pixmap.width}x{pixmap.height}px, calling GPT..."
                )

            desc = ""
            for attempt in range(2):
                try:
                    desc = call_vision(
                        page_1idx, img_idx + 1, b64_uri,
                        before, after, oai_client, paper_title,
                    )
                    break
                except Exception as exc:
                    if attempt == 0:
                        if print_progress:
                            print(f"      retry after: {exc}")
                        time.sleep(3)
                    else:
                        desc = f"ERROR: {exc}"

            if desc == "SKIP":
                if print_progress:
                    print("      -> SKIP")
                continue

            if print_progress:
                print(f"      -> {desc[:90]}...")

            image_chunks.append({
                "chunk_type":     "image",
                "parent_section": page_section,
                "source_page":    page_1idx,
                "y_order":        y_top_in,
                "content": (
                    f"[Section: {page_section}]\n"
                    f"[IMAGE page {page_1idx}]\n"
                    f"{desc}"
                ),
            })

            time.sleep(0.3)

    pdf.close()
    return image_chunks
