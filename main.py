"""
main.py — CLI pipeline orchestrator for domain-specific PDF keyword extraction.

Usage:
    python3 main.py <pdf_path> [--output output.json] [--top-k 80] [--verbose]
"""

import argparse
import os
import sys
import time
from typing import List, Dict

# Pipeline modules
from pdf_loader import load_pdf
from preprocessing import preprocess_document
from chunking import chunk_document
from llm_extractor import extract_keywords_llm
from noun_phrase_extractor import extract_noun_phrases
from acronym_merger import build_acronym_map, merge_terms_with_acronyms
from embedding_ranker import filter_by_relevance
from semantic_deduplicator import deduplicate_terms
from keyword_ranker import rank_keywords
from output_writer import write_json_output, write_excel_output, print_summary, build_page_map
import re
from config import (
    CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, MIN_CHUNK_WORDS,
    OLLAMA_MODEL, TOP_K_KEYWORDS, GENERIC_STOPWORDS,
    TARGET_KEYWORDS,
)


# Single generic words that are NOT power-sector acronyms and should be excluded
_SINGLE_WORD_BLOCKLIST = {
    "wind", "solar", "storage", "wheeling", "distribution", "scheduling",
    "licensee", "tariff", "generation", "transmission", "demand",
    "supply", "power", "energy", "electricity", "charge", "charges",
    "load", "grid", "network", "frequency", "voltage", "current",
    "capacity", "plant", "station", "utility", "consumer", "customer",
    "market", "trading", "exchange", "contract", "agreement",
    "commission", "authority", "ministry", "department", "board",
    "cost", "price", "rate", "revenue", "income", "expense",
    "asset", "liability", "capital", "debt", "equity", "investment",
    "reform", "development", "planning", "operation", "maintenance",
    "project", "scheme", "programme", "initiative", "proposal",
    "petitioner", "respondent", "appellant", "applicant",
    "regulation", "notification", "amendment", "order",
    "depreciation", "amortization", "inflation", "interest",
}


def _clean_term_text(term: str) -> str:
    """Clean a term string: remove embedded newlines, extra whitespace, noise."""
    # Replace newlines with spaces
    term = term.replace("\n", " ").replace("\r", " ")
    # Collapse whitespace
    term = " ".join(term.split())
    # Remove leading noise phrases
    term = re.sub(r"^(?:said|above|below|mentioned|concerned|further|respective|overall|applicable|net|in case the|actual value of|as per the|on the basis of|with respect to|in respect of)\s+",
                  "", term, flags=re.IGNORECASE)
    # Remove leading punctuation artifacts
    term = re.sub(r"^['\",;:()+\-–—.]+\s*", "", term)
    # Remove trailing artifacts including incomplete parentheticals
    term = re.sub(r"\s*\([^)]*$", "", term)  # Remove trailing unclosed parens
    term = re.sub(r"\s*['\",;:()+\-–—.]+$", "", term)
    # Fix concatenated words like "fifthcontrol" → skip (too noisy)
    # Remove table/figure references embedded in terms
    term = re.sub(r"\bTable\s+[IVX]+\b", "", term, flags=re.IGNORECASE)
    term = re.sub(r"\bFigure\s+\d+\b", "", term, flags=re.IGNORECASE)
    term = " ".join(term.split()).strip()
    return term


def _extract_exact_matches(full_text: str, target_keywords: List[str]) -> List[str]:
    """Find any of the 180 target keywords present in the document using regex word boundaries."""
    matches = []
    for kw in target_keywords:
        # Clean target keyword for matching (acronym parentheticals removed, e.g. "Electricity Act 2003")
        # Split acronym if in parentheses, e.g., "National Renewable Energy Policy (NREP)"
        # We want to match either the full phrase or the acronym!
        phrases_to_check = []
        paren_match = re.search(r"\(([^)]+)\)", kw)
        if paren_match:
            acronym = paren_match.group(1).strip()
            full_phrase = re.sub(r"\s*\([^)]*\)", "", kw).strip()
            phrases_to_check = [full_phrase, acronym]
        else:
            phrases_to_check = [kw.strip()]
        
        matched_any = False
        for phrase in phrases_to_check:
            # Escape regex characters and check for word boundaries
            escaped = re.escape(phrase)
            pattern = rf"\b{escaped}\b"
            if re.search(pattern, full_text, re.IGNORECASE):
                matched_any = True
                break
        
        if matched_any:
            # Add the clean canonical form of the target keyword (without parentheses)
            clean_canonical = re.sub(r"\s*\([^)]*\)", "", kw).strip()
            matches.append(clean_canonical)
            
    return matches


def _collect_all_terms(chunk_terms_llm: Dict[int, List[str]],
                       chunk_terms_np: Dict[int, List[str]]) -> List[str]:
    """Collect and deduplicate all extracted terms from both sources."""
    all_terms_lower = {}  # lower → best cased version

    for source in [chunk_terms_llm, chunk_terms_np]:
        for chunk_id, terms in source.items():
            for raw_term in terms:
                term = _clean_term_text(raw_term)
                if not term or len(term) < 2:
                    continue

                # Skip single generic stopwords
                if term.lower() in GENERIC_STOPWORDS:
                    continue

                # Skip single words unless they're known acronyms (all uppercase, 2-8 chars)
                if len(term.split()) == 1:
                    if term.lower() in _SINGLE_WORD_BLOCKLIST:
                        continue
                    # Allow only acronyms (all uppercase, 2-8 chars)
                    if not (term.isupper() and 2 <= len(term) <= 8):
                        continue

                # Skip purely numeric terms
                if term.replace(".", "").replace(",", "").replace("%", "").isdigit():
                    continue

                # Skip terms that are mostly numbers with a few words
                words = term.split()
                num_words = sum(1 for w in words if w.replace(".", "").replace(",", "").isdigit())
                if len(words) > 1 and num_words > len(words) * 0.6:
                    continue

                lower = term.lower()
                if lower not in all_terms_lower:
                    all_terms_lower[lower] = term
                else:
                    # Prefer title case or longer version
                    existing = all_terms_lower[lower]
                    if (term[0].isupper() and not existing[0].isupper()) or \
                       (len(term) > len(existing)):
                        all_terms_lower[lower] = term

    return list(all_terms_lower.values())


def run_pipeline(pdf_path: str,
                 output_path: str = "keywords_output.json",
                 excel_output: str = "",
                 top_k: int = TOP_K_KEYWORDS,
                 verbose: bool = False) -> str:
    """
    Run the full keyword extraction pipeline.

    Pipeline stages:
    1. PDF Loading
    2. Text Preprocessing
    3. Document Chunking
    4. LLM-based Keyword Extraction
    5. NP/KeyBERT Extraction
    6. Acronym Detection & Merging
    7. Embedding-based Relevance Filtering
    8. Semantic Deduplication
    9. MMR-based Final Ranking
    10. JSON Output

    Args:
        pdf_path: Path to the PDF file.
        output_path: Path for the JSON output.
        top_k: Number of top keywords to return.
        verbose: Enable verbose logging.

    Returns:
        Path to the output JSON file.
    """
    pipeline_start = time.time()
    print("=" * 70)
    print("  DOMAIN-SPECIFIC PDF KEYWORD EXTRACTION SYSTEM")
    print("  Indian Power Sector / Electricity Markets / Regulatory Policy")
    print("=" * 70)
    print()

    # ── Stage 1: PDF Loading ──────────────────────────────────
    stage_start = time.time()
    print("▶ Stage 1/10: Loading PDF...")
    doc = load_pdf(pdf_path)
    print(f"  Title: {doc.title}")
    print(f"  ⏱ {time.time() - stage_start:.1f}s\n")

    # ── Stage 2: Preprocessing ────────────────────────────────
    stage_start = time.time()
    print("▶ Stage 2/10: Preprocessing text...")
    doc = preprocess_document(doc)
    print(f"  ⏱ {time.time() - stage_start:.1f}s\n")

    # ── Stage 3: Chunking ─────────────────────────────────────
    stage_start = time.time()
    print("▶ Stage 3/10: Chunking document...")
    chunks = chunk_document(
        doc,
        target_words=CHUNK_SIZE_WORDS,
        overlap_words=CHUNK_OVERLAP_WORDS,
        min_chunk_words=MIN_CHUNK_WORDS,
    )
    if verbose:
        for chunk in chunks:
            print(f"  {chunk}")
    print(f"  ⏱ {time.time() - stage_start:.1f}s\n")

    # ── Stage 4: LLM Extraction ──────────────────────────────
    stage_start = time.time()
    print(f"▶ Stage 4/10: LLM extraction ({OLLAMA_MODEL})...")
    chunk_terms_llm = extract_keywords_llm(chunks, model=OLLAMA_MODEL)
    llm_unique = set()
    for terms in chunk_terms_llm.values():
        llm_unique.update(t.lower() for t in terms)
    print(f"  LLM extracted {len(llm_unique)} unique terms")
    print(f"  ⏱ {time.time() - stage_start:.1f}s\n")

    # ── Stage 5: NP/KeyBERT Extraction ───────────────────────
    stage_start = time.time()
    print("▶ Stage 5/10: NP/KeyBERT extraction...")
    chunk_terms_np = extract_noun_phrases(chunks)
    np_unique = set()
    for terms in chunk_terms_np.values():
        np_unique.update(t.lower() for t in terms)
    print(f"  NP extracted {len(np_unique)} unique terms")
    print(f"  ⏱ {time.time() - stage_start:.1f}s\n")

    # ── Stage 6: Acronym Detection ────────────────────────────
    stage_start = time.time()
    print("▶ Stage 6/10: Detecting acronyms...")
    acr_map = build_acronym_map(doc.full_text)
    print(f"  ⏱ {time.time() - stage_start:.1f}s\n")

    # ── Stage 6.5: Exact Dictionary Matching ──────────────────
    stage_start_exact = time.time()
    print("▶ Stage 6.5/10: Exact dictionary matching...")
    exact_matches = _extract_exact_matches(doc.full_text, TARGET_KEYWORDS)
    print(f"  Found {len(exact_matches)} target keywords in document")
    print(f"  ⏱ {time.time() - stage_start_exact:.1f}s\n")

    # ── Stage 7: Collect & Filter ─────────────────────────────
    stage_start = time.time()
    print("▶ Stage 7/10: Collecting and filtering terms...")
    all_terms = _collect_all_terms(chunk_terms_llm, chunk_terms_np)
    # Inject exact-matched terms to ensure they are candidates
    all_terms = list(set(all_terms + exact_matches))
    print(f"  Collected {len(all_terms)} unique candidate terms")

    # Apply acronym merging to get canonical forms
    merged = merge_terms_with_acronyms(all_terms, acr_map)
    canonical_terms = [m["canonical"] for m in merged]
    print(f"  After acronym merge: {len(canonical_terms)} terms")

    # Embedding-based relevance filtering
    filtered = filter_by_relevance(canonical_terms)
    filtered_terms = [t for t, s in filtered]
    filtered_scores = {t: s for t, s in filtered}
    
    # Guarantee that ALL exact-matched target terms present in the document
    # are kept and score 1.0 (or higher)
    exact_lower_set = {t.lower() for t in exact_matches}
    for term in canonical_terms:
        if term.lower() in exact_lower_set:
            if term not in filtered_scores:
                filtered_terms.append(term)
                filtered_scores[term] = 1.0
            else:
                filtered_scores[term] = max(filtered_scores[term], 1.0)
                
    print(f"  After relevance filter: {len(filtered_terms)} terms")
    print(f"  ⏱ {time.time() - stage_start:.1f}s\n")

    # ── Stage 8: Semantic Deduplication ───────────────────────
    stage_start = time.time()
    print("▶ Stage 8/10: Semantic deduplication...")
    deduped = deduplicate_terms(filtered_terms, filtered_scores)

    # Merge acronym aliases into deduped results
    alias_lookup = {m["canonical"].lower(): m["aliases"] for m in merged}
    for entry in deduped:
        canonical_lower = entry["canonical"].lower()
        acr_aliases = alias_lookup.get(canonical_lower, [])
        existing_aliases = set(entry.get("aliases", []))
        existing_aliases.update(acr_aliases)
        # Remove self-references
        existing_aliases.discard(entry["canonical"])
        entry["aliases"] = list(existing_aliases)

    print(f"  ⏱ {time.time() - stage_start:.1f}s\n")

    # ── Stage 9: Final Ranking ────────────────────────────────
    stage_start = time.time()
    print(f"▶ Stage 9/10: Final ranking (top {top_k})...")
    ranked = rank_keywords(
        deduped,
        chunk_terms_llm,
        chunk_terms_np,
        total_chunks=len(chunks),
        top_k=top_k,
    )
    print(f"  ⏱ {time.time() - stage_start:.1f}s\n")

    # ── Stage 10: Output ──────────────────────────────────────
    stage_start = time.time()
    print("▶ Stage 10/10: Writing output...")

    # Build page map
    page_map = build_page_map(chunks, chunk_terms_llm, chunk_terms_np)

    # Write JSON
    write_json_output(
        keywords=ranked,
        document_title=doc.title,
        total_pages=doc.total_pages,
        page_map=page_map,
        output_path=output_path,
    )

    # Write Excel
    excel_path = excel_output or os.path.splitext(output_path)[0] + ".xlsx"
    write_excel_output(
        keywords=ranked,
        document_title=doc.title,
        total_pages=doc.total_pages,
        page_map=page_map,
        output_path=excel_path,
    )

    # Print human-readable summary
    print_summary(ranked, doc.title, doc.total_pages)

    total_time = time.time() - pipeline_start
    print(f"\n✅ Pipeline complete in {total_time:.1f}s")
    print(f"   JSON:  {os.path.abspath(output_path)}")
    print(f"   Excel: {os.path.abspath(excel_path)}")

    return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Domain-Specific PDF Keyword Extraction System "
                    "(Indian Power Sector)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py document.pdf
  python3 main.py document.pdf --output results.json --top-k 100
  python3 main.py document.pdf --verbose
        """
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF document to process"
    )
    parser.add_argument(
        "--output", "-o",
        default="keywords_output.json",
        help="Output JSON file path (default: keywords_output.json)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=TOP_K_KEYWORDS,
        help=f"Number of top keywords to extract (default: {TOP_K_KEYWORDS})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--excel", "-e",
        default="",
        help="Excel output file path (default: same name as JSON with .xlsx extension)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: File not found: {args.pdf_path}")
        sys.exit(1)

    try:
        run_pipeline(
            pdf_path=args.pdf_path,
            output_path=args.output,
            excel_output=args.excel,
            top_k=args.top_k,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
