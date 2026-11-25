"""
preprocessing.py — Clean extracted PDF text.

Handles: header/footer removal, page number stripping, boilerplate detection,
whitespace normalization, and hyphenation repair.
"""

import re
from collections import Counter
from typing import List, Dict, Set

from pdf_loader import DocumentData, PageText


def _detect_repeated_lines(pages: List[PageText], max_lines: int = 3,
                           min_occurrences_ratio: float = 0.4) -> Set[str]:
    """
    Detect lines that repeat across many pages (headers/footers).

    Looks at the first and last `max_lines` lines of each page.
    Lines appearing in more than `min_occurrences_ratio` of pages are headers/footers.
    """
    min_pages = max(3, int(len(pages) * min_occurrences_ratio))
    line_counter = Counter()

    for page in pages:
        lines = [l.strip() for l in page.raw_text.split("\n") if l.strip()]
        if len(lines) < 2:
            continue

        # Check top and bottom lines
        candidates = set()
        for line in lines[:max_lines]:
            candidates.add(line)
        for line in lines[-max_lines:]:
            candidates.add(line)

        for line in candidates:
            # Normalize: remove page numbers from the line for matching
            normalized = re.sub(r"\b\d{1,4}\b", "##", line).strip()
            if len(normalized) > 3:  # Ignore very short fragments
                line_counter[normalized] += 1

    # Lines that appear on many pages are headers/footers
    repeated = set()
    for normalized_line, count in line_counter.items():
        if count >= min_pages:
            repeated.add(normalized_line)

    return repeated


def _is_page_number_line(line: str) -> bool:
    """Check if a line is just a page number."""
    line = line.strip()
    # Pure number
    if re.match(r"^\d{1,4}$", line):
        return True
    # "Page N" or "Page N of M"
    if re.match(r"^page\s+\d+(\s+of\s+\d+)?$", line, re.IGNORECASE):
        return True
    # "- N -" or "— N —"
    if re.match(r"^[-—]\s*\d{1,4}\s*[-—]$", line):
        return True
    return False


def _remove_boilerplate(text: str) -> str:
    """Remove common boilerplate patterns."""
    # Remove "Confidential" / "Draft" watermarks
    text = re.sub(r"(?i)\b(confidential|draft|privileged)\b", "", text)
    # Remove repeated separator lines
    text = re.sub(r"[=\-_]{10,}", "", text)
    # Remove "Source:" standalone lines
    text = re.sub(r"(?m)^Source:\s*$", "", text)
    return text


def _fix_hyphenation(text: str) -> str:
    """Rejoin words broken by line-end hyphenation."""
    # Pattern: word- \n word → joined word
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return text


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    # Join lines that were split mid-sentence (line doesn't end with sentence-ending punct)
    lines = text.split("\n")
    merged_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            merged_lines.append("")  # Preserve paragraph breaks
            continue
        # If previous non-empty line doesn't end with sentence-ending punctuation,
        # and this line starts with a lowercase letter or continues a phrase, join them.
        if (merged_lines and merged_lines[-1] and
            not merged_lines[-1].rstrip().endswith(('.', ':', ';', '!', '?', ')')) and
            stripped and (stripped[0].islower() or stripped[0] in '(')):
            merged_lines[-1] = merged_lines[-1].rstrip() + " " + stripped
        else:
            merged_lines.append(stripped)
    text = "\n".join(merged_lines)
    # Collapse multiple spaces to single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse more than 2 consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace from each line
    lines = [l.strip() for l in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def clean_page(page_text: str, repeated_patterns: Set[str]) -> str:
    """Clean a single page's text."""
    lines = page_text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines (preserve structure with a blank)
        if not stripped:
            cleaned_lines.append("")
            continue

        # Skip page number lines
        if _is_page_number_line(stripped):
            continue

        # Skip lines matching detected headers/footers
        normalized = re.sub(r"\b\d{1,4}\b", "##", stripped).strip()
        if normalized in repeated_patterns:
            continue

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = _remove_boilerplate(text)
    text = _fix_hyphenation(text)
    text = _normalize_whitespace(text)
    return text


def preprocess_document(doc: DocumentData) -> DocumentData:
    """
    Clean all pages in a document.

    - Detects and removes repeated headers/footers
    - Strips page numbers
    - Removes boilerplate
    - Fixes hyphenation
    - Normalizes whitespace

    Args:
        doc: Raw DocumentData from pdf_loader.

    Returns:
        New DocumentData with cleaned page texts.
    """
    print(f"[Preprocessing] Cleaning {doc.total_pages} pages...")

    # Detect repeated patterns across pages
    repeated = _detect_repeated_lines(doc.pages)
    if repeated:
        print(f"[Preprocessing] Detected {len(repeated)} repeated header/footer patterns")

    # Clean each page
    cleaned_pages = []
    total_removed_chars = 0

    for page in doc.pages:
        original_len = len(page.raw_text)
        cleaned_text = clean_page(page.raw_text, repeated)
        cleaned_pages.append(PageText(
            page_number=page.page_number,
            raw_text=cleaned_text,
        ))
        total_removed_chars += original_len - len(cleaned_text)

    result = DocumentData(
        file_path=doc.file_path,
        title=doc.title,
        total_pages=doc.total_pages,
        pages=cleaned_pages,
    )

    total_chars = sum(p.char_count for p in cleaned_pages)
    print(f"[Preprocessing] Removed {total_removed_chars:,} chars of noise. "
          f"Clean text: {total_chars:,} chars")

    return result
