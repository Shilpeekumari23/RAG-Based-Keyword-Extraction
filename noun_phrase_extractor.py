"""
noun_phrase_extractor.py — Statistical NP extraction using spaCy + KeyBERT.

Complementary to LLM extraction: catches terms the LLM may miss
using linguistic patterns and embedding-based keyphrase extraction.
"""

import re
from typing import List, Dict, Set

from chunking import TextChunk
from config import GENERIC_STOPWORDS, EMBEDDING_MODEL


def _extract_spacy_noun_phrases(text: str, nlp) -> List[str]:
    """Extract noun phrases using spaCy's dependency parser."""
    # Process in chunks to avoid memory issues with large texts
    max_chars = 100000
    if len(text) > max_chars:
        text = text[:max_chars]

    doc = nlp(text)
    phrases = []

    for np in doc.noun_chunks:
        # Get the full noun phrase text
        phrase = np.text.strip()

        # Filter: keep only 2-6 word phrases
        words = phrase.split()
        if len(words) < 2 or len(words) > 6:
            continue

        # Filter: skip if all words are generic stopwords
        meaningful_words = [w.lower() for w in words if w.lower() not in GENERIC_STOPWORDS]
        if len(meaningful_words) < 1:
            continue

        # Filter: skip phrases starting with determiners/pronouns
        if words[0].lower() in {"the", "a", "an", "this", "that", "these", "those",
                                  "its", "their", "our", "his", "her", "my", "your",
                                  "some", "any", "all", "each", "every", "such", "same"}:
            # Remove the determiner
            phrase = " ".join(words[1:])
            if len(phrase.split()) < 2:
                continue

        # Filter: skip numeric-only phrases
        if re.match(r"^[\d\s.,%-]+$", phrase):
            continue

        phrases.append(phrase)

    return phrases


def _extract_regex_patterns(text: str) -> List[str]:
    """Extract domain-specific patterns using regex."""
    patterns = []

    # Acronym with expansion: "Full Name (ACRONYM)" or "ACRONYM (Full Name)"
    for match in re.finditer(r"([A-Z][a-zA-Z\s&]{5,50})\s*\(([A-Z]{2,8})\)", text):
        expansion, acronym = match.group(1).strip(), match.group(2)
        patterns.append(expansion)
        patterns.append(acronym)

    for match in re.finditer(r"([A-Z]{2,8})\s*\(([A-Z][a-zA-Z\s&]{5,50})\)", text):
        acronym, expansion = match.group(1), match.group(2).strip()
        patterns.append(expansion)
        patterns.append(acronym)

    # Standalone acronyms (2-8 uppercase letters)
    for match in re.finditer(r"\b([A-Z]{2,8})\b", text):
        acr = match.group(1)
        # Filter out common non-acronym uppercase words
        if acr not in {"AND", "THE", "FOR", "NOT", "BUT", "NOR", "YET", "ARE",
                       "WAS", "HAS", "HAD", "MAY", "CAN", "SHALL", "WILL",
                       "WITH", "FROM", "THAT", "THIS", "BEEN", "HAVE", "ALSO",
                       "SUCH", "INTO", "OVER", "UPON", "THAN", "THEM", "THEN",
                       "ONLY", "EACH", "ALL", "ANY", "FIG", "TAB", "REF",
                       "VOL", "SEC", "II", "III", "IV", "VI", "VII", "VIII",
                       "IX", "XI", "XII"}:
            patterns.append(acr)

    # Section/Regulation references: "Section 62", "Regulation 6.4"
    for match in re.finditer(r"(Section|Regulation|Clause|Rule|Schedule)\s+\d+[\d.]*", text):
        patterns.append(match.group(0))

    return patterns


def _extract_keybert_phrases(text: str, top_n: int = 30) -> List[str]:
    """Extract keyphrases using KeyBERT with domain-aware embeddings."""
    try:
        from keybert import KeyBERT

        # Truncate text for KeyBERT
        words = text.split()
        if len(words) > 1500:
            text = " ".join(words[:1500])

        kw_model = KeyBERT(model=EMBEDDING_MODEL)

        # Extract with different n-gram ranges
        keywords_2 = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(2, 4),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,
            diversity=0.5,
        )
        keywords_1 = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),
            stop_words="english",
            top_n=10,
            use_mmr=True,
            diversity=0.5,
        )

        # Combine, filter by score
        all_kw = keywords_2 + keywords_1
        phrases = []
        for kw, score in all_kw:
            if score > 0.2:  # Minimum relevance threshold
                kw_clean = kw.strip()
                # Skip single generic words
                if len(kw_clean.split()) == 1 and kw_clean.lower() in GENERIC_STOPWORDS:
                    continue
                phrases.append(kw_clean)

        return phrases

    except ImportError:
        print("  [NP Extractor] KeyBERT not available, skipping.")
        return []
    except Exception as e:
        print(f"  [NP Extractor] KeyBERT error: {e}")
        return []


def extract_noun_phrases(chunks: List[TextChunk]) -> Dict[int, List[str]]:
    """
    Extract noun phrases and keyphrases from all chunks.

    Combines:
    1. spaCy noun phrase extraction
    2. Regex pattern extraction (acronyms, section refs)
    3. KeyBERT embedding-based keyphrase extraction

    Args:
        chunks: List of TextChunks.

    Returns:
        Dict mapping chunk_id → list of extracted terms.
    """
    print(f"[NP Extractor] Processing {len(chunks)} chunks...")

    # Load spaCy model
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        nlp.max_length = 200000
        has_spacy = True
        print("  [NP Extractor] spaCy model loaded")
    except (ImportError, OSError):
        print("  [NP Extractor] spaCy not available, using regex only")
        has_spacy = False
        nlp = None

    results = {}
    all_terms = set()

    for i, chunk in enumerate(chunks):
        terms = []

        # 1. spaCy noun phrases
        if has_spacy:
            spacy_terms = _extract_spacy_noun_phrases(chunk.text, nlp)
            terms.extend(spacy_terms)

        # 2. Regex patterns (always available)
        regex_terms = _extract_regex_patterns(chunk.text)
        terms.extend(regex_terms)

        # 3. KeyBERT (on every 3rd chunk to save time, plus first and last)
        if i == 0 or i == len(chunks) - 1 or i % 3 == 0:
            kb_terms = _extract_keybert_phrases(chunk.text, top_n=20)
            terms.extend(kb_terms)

        # Deduplicate within chunk (case-insensitive)
        seen = set()
        unique_terms = []
        for t in terms:
            key = t.lower().strip()
            if key not in seen and len(key) > 2:
                seen.add(key)
                unique_terms.append(t.strip())

        results[chunk.chunk_id] = unique_terms
        all_terms.update(t.lower() for t in unique_terms)

    print(f"[NP Extractor] Done. Total unique terms: {len(all_terms)}")
    return results
