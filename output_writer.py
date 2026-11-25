"""
output_writer.py — Write extraction results to structured JSON.

Produces the final JSON output matching the specified schema,
plus a human-readable console summary.
"""

import json
import os
from datetime import datetime
from typing import List, Dict

from config import OLLAMA_MODEL


def write_json_output(keywords: List[Dict],
                      document_title: str,
                      total_pages: int,
                      page_map: Dict[str, List[int]],
                      output_path: str) -> str:
    """
    Write the final keyword extraction results to a JSON file.

    Args:
        keywords: List of ranked keyword dicts from keyword_ranker.
        document_title: Title of the source document.
        total_pages: Total number of pages in the document.
        page_map: Dict mapping term (lowercase) → list of page numbers.
        output_path: Path to write the JSON output.

    Returns:
        Path to the written file.
    """
    # Build output structure
    output = {
        "document_title": document_title,
        "total_pages": total_pages,
        "extraction_timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_used": OLLAMA_MODEL,
        "total_keywords": len(keywords),
        "keywords": []
    }

    for kw in keywords:
        term = kw["term"]
        term_lower = term.lower()

        # Get page numbers for this term and its aliases
        pages = set(page_map.get(term_lower, []))
        for alias in kw.get("aliases", []):
            pages.update(page_map.get(alias.lower(), []))
        pages = sorted(pages)

        output["keywords"].append({
            "term": term,
            "score": kw["score"],
            "category": kw["category"],
            "pages": pages,
            "aliases": kw.get("aliases", []),
        })

    # Write JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[Output] JSON written to: {output_path}")
    return output_path


def write_excel_output(keywords: List[Dict],
                       document_title: str,
                       total_pages: int,
                       page_map: Dict[str, List[int]],
                       output_path: str) -> str:
    """
    Write keyword extraction results to a formatted Excel file.

    Args:
        keywords: List of ranked keyword dicts from keyword_ranker.
        document_title: Title of the source document.
        total_pages: Total number of pages in the document.
        page_map: Dict mapping term (lowercase) → list of page numbers.
        output_path: Path to write the Excel file (.xlsx).

    Returns:
        Path to the written file.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("[WARNING] openpyxl not installed. Skipping Excel output.")
        print("  Install with: pip3 install --user openpyxl")
        return ""

    wb = Workbook()

    # ── Sheet 1: Keywords ─────────────────────────────────────
    ws = wb.active
    ws.title = "Keywords"

    # Category colors
    category_colors = {
        "Regulatory": "FFE4CC",        # light orange
        "Market": "CCEBFF",            # light blue
        "Grid Operations": "D4EDDA",   # light green
        "Transmission & Distribution": "FFF3CD",  # light yellow
        "Renewable Energy": "D1ECF1",  # light cyan
        "Economic & Financial": "E2D9F3",  # light purple
        "Policy & Climate": "F8D7DA",  # light red
    }

    # Styles
    header_font = Font(name="Calibri", bold=True, size=11, color="FFFFFF")
    header_fill = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell_align = Alignment(vertical="top", wrap_text=True)
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    # Headers
    headers = ["Rank", "Term", "Score", "Category", "Aliases", "Pages", "Page Count"]
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = thin_border

    # Data rows
    for i, kw in enumerate(keywords):
        row = i + 2
        term = kw["term"]
        term_lower = term.lower()

        # Get pages
        pages = set(page_map.get(term_lower, []))
        for alias in kw.get("aliases", []):
            pages.update(page_map.get(alias.lower(), []))
        pages = sorted(pages)
        pages_str = ", ".join(str(p) for p in pages)

        # Write cells
        ws.cell(row=row, column=1, value=i + 1).alignment = Alignment(horizontal="center")
        ws.cell(row=row, column=2, value=term)
        ws.cell(row=row, column=3, value=round(kw["score"], 4))
        ws.cell(row=row, column=4, value=kw["category"])
        ws.cell(row=row, column=5, value=", ".join(kw.get("aliases", [])))
        ws.cell(row=row, column=6, value=pages_str)
        ws.cell(row=row, column=7, value=len(pages))

        # Apply category color
        cat_color = category_colors.get(kw["category"], "FFFFFF")
        cat_fill = PatternFill(start_color=cat_color, end_color=cat_color, fill_type="solid")

        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=row, column=col)
            cell.alignment = cell_align
            cell.border = thin_border
            cell.font = Font(name="Calibri", size=10)
            if col == 4:  # Category column
                cell.fill = cat_fill

    # Column widths
    col_widths = [6, 42, 8, 28, 45, 40, 10]
    for i, width in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = width

    # Auto-filter
    ws.auto_filter.ref = f"A1:G{len(keywords) + 1}"

    # Freeze top row
    ws.freeze_panes = "A2"

    # ── Sheet 2: Summary ──────────────────────────────────────
    ws2 = wb.create_sheet(title="Summary")
    ws2.column_dimensions["A"].width = 30
    ws2.column_dimensions["B"].width = 40

    summary_data = [
        ("Document Title", document_title),
        ("Total Pages", total_pages),
        ("Total Keywords", len(keywords)),
        ("Extraction Date", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Model Used", OLLAMA_MODEL),
        ("", ""),
        ("Category", "Count"),
    ]

    # Count by category
    cat_counts = {}
    for kw in keywords:
        cat = kw["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat in sorted(cat_counts.keys()):
        summary_data.append((cat, cat_counts[cat]))

    for row_idx, (label, value) in enumerate(summary_data, 1):
        cell_a = ws2.cell(row=row_idx, column=1, value=label)
        cell_b = ws2.cell(row=row_idx, column=2, value=value)
        cell_a.font = Font(name="Calibri", bold=True, size=11)
        cell_b.font = Font(name="Calibri", size=11)
        if row_idx == 7:  # "Category" header row
            cell_a.font = Font(name="Calibri", bold=True, size=11, color="FFFFFF")
            cell_b.font = Font(name="Calibri", bold=True, size=11, color="FFFFFF")
            cell_a.fill = header_fill
            cell_b.fill = header_fill

    # Save
    wb.save(output_path)
    print(f"[Output] Excel written to: {output_path}")
    return output_path


def print_summary(keywords: List[Dict], document_title: str, total_pages: int):
    """Print a human-readable summary to console."""
    print("\n" + "=" * 70)
    print(f"  KEYWORD EXTRACTION RESULTS")
    print(f"  Document: {document_title}")
    print(f"  Pages: {total_pages}")
    print(f"  Keywords extracted: {len(keywords)}")
    print("=" * 70)

    # Group by category
    categories = {}
    for kw in keywords:
        cat = kw["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(kw)

    for category in sorted(categories.keys()):
        cat_kws = categories[category]
        print(f"\n  📂 {category} ({len(cat_kws)} terms)")
        print(f"  {'─' * 40}")
        for kw in cat_kws[:10]:  # Show top 10 per category
            aliases_str = ""
            if kw.get("aliases"):
                aliases_str = f" (aka: {', '.join(kw['aliases'][:3])})"
            print(f"    • {kw['term']} [{kw['score']:.3f}]{aliases_str}")
        if len(cat_kws) > 10:
            print(f"    ... and {len(cat_kws) - 10} more")

    print("\n" + "=" * 70)


def build_page_map(chunks, chunk_terms_llm: Dict[int, List[str]],
                   chunk_terms_np: Dict[int, List[str]]) -> Dict[str, List[int]]:
    """
    Build a mapping from term (lowercase) → list of page numbers where it appears.

    Args:
        chunks: List of TextChunk objects.
        chunk_terms_llm: LLM-extracted terms per chunk.
        chunk_terms_np: NP-extracted terms per chunk.

    Returns:
        Dict mapping term_lower → sorted list of page numbers.
    """
    # Build chunk_id → page range mapping
    chunk_pages = {}
    for chunk in chunks:
        pages = list(range(chunk.page_start, chunk.page_end + 1))
        chunk_pages[chunk.chunk_id] = pages

    # Map terms to pages
    page_map = {}
    for source in [chunk_terms_llm, chunk_terms_np]:
        for chunk_id, terms in source.items():
            pages = chunk_pages.get(chunk_id, [])
            for term in terms:
                term_lower = term.lower().strip()
                if term_lower not in page_map:
                    page_map[term_lower] = set()
                page_map[term_lower].update(pages)

    # Convert sets to sorted lists
    return {term: sorted(pages) for term, pages in page_map.items()}
