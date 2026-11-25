"""
chunking.py — Split cleaned document text into semantically meaningful chunks.

Target: 1000–1500 words per chunk with 200-word overlap.
Splits on paragraph boundaries and section headings.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from pdf_loader import DocumentData


@dataclass
class TextChunk:
    """A chunk of document text with metadata."""
    chunk_id: int
    text: str
    word_count: int
    page_start: int          # First page in this chunk
    page_end: int            # Last page in this chunk
    section_heading: str = ""  # Detected section heading, if any

    def __repr__(self):
        return (f"Chunk({self.chunk_id}, pages={self.page_start}-{self.page_end}, "
                f"words={self.word_count}, section='{self.section_heading[:40]}')")


def _detect_section_heading(text: str) -> str:
    """Try to detect a section heading from the beginning of text."""
    lines = text.strip().split("\n")
    for line in lines[:3]:
        line = line.strip()
        if not line:
            continue
        # Numbered section heading: "1.", "1.1", "Chapter 1", "Section 2.3"
        if re.match(r"^(\d+\.?\d*\.?\d*\s+|chapter\s+\d+|section\s+\d+)", line, re.IGNORECASE):
            return line[:120]
        # ALL CAPS heading (likely a section title)
        if line.isupper() and 5 < len(line) < 120:
            return line
        # Title Case heading that's not too long
        if line.istitle() and 5 < len(line) < 120 and not line.endswith("."):
            return line
    return ""


def _build_page_word_map(doc: DocumentData) -> List[Tuple[int, str]]:
    """
    Build a flat list of (page_number, paragraph_text) tuples preserving order.
    """
    result = []
    for page in doc.pages:
        text = page.raw_text.strip()
        if not text:
            continue
        # Split into paragraphs (separated by blank lines)
        paragraphs = re.split(r"\n\s*\n", text)
        for para in paragraphs:
            para = para.strip()
            if para:
                result.append((page.page_number, para))
    return result


def chunk_document(doc: DocumentData,
                   target_words: int = 1200,
                   overlap_words: int = 200,
                   min_chunk_words: int = 300) -> List[TextChunk]:
    """
    Split a document into overlapping chunks of approximately `target_words` words.

    Strategy:
    - Build a flat paragraph list with page tracking
    - Greedily accumulate paragraphs until target size
    - Add overlap from previous chunk's tail
    - Detect section headings for each chunk

    Args:
        doc: Preprocessed DocumentData.
        target_words: Target number of words per chunk.
        overlap_words: Number of words to overlap between chunks.
        min_chunk_words: Minimum chunk size (don't create tiny final chunks).

    Returns:
        List of TextChunk objects.
    """
    paragraphs = _build_page_word_map(doc)

    if not paragraphs:
        print("[Chunking] WARNING: No text to chunk.")
        return []

    chunks = []
    chunk_id = 0
    para_idx = 0
    overlap_text = ""  # Overlap from previous chunk

    while para_idx < len(paragraphs):
        # Start with overlap from previous chunk
        current_paragraphs = []
        current_pages = set()
        current_word_count = 0

        # Add overlap text if available
        if overlap_text:
            current_paragraphs.append(overlap_text)
            current_word_count += len(overlap_text.split())

        # Accumulate paragraphs until target size
        chunk_start_idx = para_idx
        while para_idx < len(paragraphs) and current_word_count < target_words:
            page_num, para_text = paragraphs[para_idx]
            current_paragraphs.append(para_text)
            current_pages.add(page_num)
            current_word_count += len(para_text.split())
            para_idx += 1

        # If this would be a tiny final chunk, merge with previous
        if current_word_count < min_chunk_words and chunks:
            # Append to previous chunk
            prev = chunks[-1]
            merged_text = prev.text + "\n\n" + "\n\n".join(current_paragraphs)
            chunks[-1] = TextChunk(
                chunk_id=prev.chunk_id,
                text=merged_text,
                word_count=len(merged_text.split()),
                page_start=prev.page_start,
                page_end=max(current_pages) if current_pages else prev.page_end,
                section_heading=prev.section_heading,
            )
            continue

        # Build chunk text
        chunk_text = "\n\n".join(current_paragraphs)
        page_list = sorted(current_pages) if current_pages else [1]

        chunk = TextChunk(
            chunk_id=chunk_id,
            text=chunk_text,
            word_count=len(chunk_text.split()),
            page_start=page_list[0],
            page_end=page_list[-1],
            section_heading=_detect_section_heading(chunk_text),
        )
        chunks.append(chunk)
        chunk_id += 1

        # Prepare overlap for next chunk: take last `overlap_words` words
        words = chunk_text.split()
        if len(words) > overlap_words:
            overlap_text = " ".join(words[-overlap_words:])
        else:
            overlap_text = ""

    total_words = sum(c.word_count for c in chunks)
    print(f"[Chunking] Created {len(chunks)} chunks, "
          f"avg {total_words // max(len(chunks), 1)} words/chunk, "
          f"total {total_words:,} words")

    return chunks
