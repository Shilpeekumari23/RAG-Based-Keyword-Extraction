"""
pdf_loader.py — Extract text from PDF documents with page-level tracking.

Primary: pdfplumber (table-aware, preserves layout).
Fallback: PyMuPDF (faster, better on scanned-like PDFs).
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PageText:
    """Container for a single page's extracted text."""
    page_number: int          # 1-indexed
    raw_text: str
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.raw_text)


@dataclass
class DocumentData:
    """Container for an entire document's extracted content."""
    file_path: str
    title: str
    total_pages: int
    pages: List[PageText] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.raw_text for p in self.pages if p.raw_text.strip())

    @property
    def non_empty_pages(self) -> int:
        return sum(1 for p in self.pages if p.raw_text.strip())


def _extract_title_from_text(first_page_text: str, file_path: str) -> str:
    """Attempt to extract a document title from the first page text or filename."""
    lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]

    # Strategy 1: First substantial line (likely a title)
    for line in lines[:10]:
        # Skip very short lines (page numbers, dates) and very long lines (paragraphs)
        if 10 < len(line) < 200:
            # Skip lines that look like dates, page numbers, or headers
            if re.match(r"^\d+$", line):
                continue
            if re.match(r"^(page|dated|date|no\.|sr\.|sl\.)", line, re.IGNORECASE):
                continue
            return line

    # Strategy 2: Derive from filename
    basename = os.path.splitext(os.path.basename(file_path))[0]
    # Clean up common filename patterns
    title = re.sub(r"[_,]", " ", basename)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def load_pdf_pdfplumber(file_path: str) -> Optional[DocumentData]:
    """Extract text using pdfplumber (primary method)."""
    try:
        import pdfplumber
    except ImportError:
        print("[WARNING] pdfplumber not installed. Trying fallback.")
        return None

    try:
        pages = []
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(PageText(page_number=i + 1, raw_text=text))

        if not pages:
            return None

        title = _extract_title_from_text(pages[0].raw_text, file_path)
        doc = DocumentData(
            file_path=file_path,
            title=title,
            total_pages=total_pages,
            pages=pages,
        )

        print(f"[PDF Loader] pdfplumber: {total_pages} pages, "
              f"{doc.non_empty_pages} with text, "
              f"{sum(p.char_count for p in pages):,} chars total")
        return doc

    except Exception as e:
        print(f"[WARNING] pdfplumber failed: {e}")
        return None


def load_pdf_pymupdf(file_path: str) -> Optional[DocumentData]:
    """Extract text using PyMuPDF / fitz (fallback method)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[ERROR] Neither pdfplumber nor PyMuPDF available.")
        return None

    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        pages = []

        for i in range(total_pages):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            pages.append(PageText(page_number=i + 1, raw_text=text))

        doc.close()

        if not pages:
            return None

        title = _extract_title_from_text(pages[0].raw_text, file_path)
        doc_data = DocumentData(
            file_path=file_path,
            title=title,
            total_pages=total_pages,
            pages=pages,
        )

        print(f"[PDF Loader] PyMuPDF: {total_pages} pages, "
              f"{doc_data.non_empty_pages} with text, "
              f"{sum(p.char_count for p in pages):,} chars total")
        return doc_data

    except Exception as e:
        print(f"[ERROR] PyMuPDF failed: {e}")
        return None


def load_pdf(file_path: str) -> DocumentData:
    """
    Load a PDF document, trying pdfplumber first, then PyMuPDF as fallback.

    Args:
        file_path: Path to the PDF file.

    Returns:
        DocumentData with page-level text extraction.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        RuntimeError: If both extraction methods fail.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Not a PDF file: {file_path}")

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"[PDF Loader] Loading: {os.path.basename(file_path)} ({file_size_mb:.1f} MB)")

    # Try pdfplumber first
    result = load_pdf_pdfplumber(file_path)

    # Fall back to PyMuPDF if pdfplumber fails or yields very little text
    if result is None or result.non_empty_pages < result.total_pages * 0.5:
        if result is not None:
            print(f"[PDF Loader] pdfplumber yielded sparse results, trying PyMuPDF...")
        fallback = load_pdf_pymupdf(file_path)
        if fallback is not None:
            # Use fallback if it got more text
            if result is None or fallback.non_empty_pages > result.non_empty_pages:
                result = fallback

    if result is None:
        raise RuntimeError(f"Failed to extract text from: {file_path}")

    return result
