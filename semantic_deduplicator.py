"""
semantic_deduplicator.py — Remove near-duplicate terms using embedding clustering.

Groups similar terms (e.g., "Renewable Purchase Obligation" and
"Renewable Purchase Obligations") and selects the best canonical form.
"""

import re
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import AgglomerativeClustering

from config import DEDUP_SIMILARITY_THRESHOLD
from embedding_ranker import encode_terms


def _select_canonical(cluster_terms: List[str],
                      term_scores: Dict[str, float]) -> Tuple[str, List[str]]:
    """
    Select the best canonical term from a cluster.

    Priority:
    1. Highest domain relevance score
    2. Longest (most descriptive) form
    3. Title case over lowercase
    """
    if len(cluster_terms) == 1:
        return cluster_terms[0], []

    # Score each candidate
    scored = []
    for term in cluster_terms:
        relevance = term_scores.get(term, 0.0)
        # Prefer longer, more descriptive terms
        length_bonus = min(len(term.split()) / 6.0, 1.0) * 0.1
        # Prefer title case
        case_bonus = 0.05 if (term[0].isupper() and not term.isupper()) else 0.0
        # Penalize pure acronyms as canonical (they become aliases)
        acronym_penalty = -0.2 if (term.isupper() and len(term) <= 8) else 0.0

        total = relevance + length_bonus + case_bonus + acronym_penalty
        scored.append((term, total))

    scored.sort(key=lambda x: x[1], reverse=True)

    canonical = scored[0][0]
    aliases = [t for t, _ in scored[1:] if t != canonical]

    return canonical, aliases


def deduplicate_terms(terms: List[str],
                      term_scores: Dict[str, float],
                      threshold: float = DEDUP_SIMILARITY_THRESHOLD
                      ) -> List[Dict]:
    """
    Deduplicate terms using embedding-based clustering.

    Terms with cosine similarity >= threshold are grouped into clusters.
    The best representative is selected as canonical; others become aliases.

    Args:
        terms: List of candidate terms.
        term_scores: Dict mapping term → relevance score.
        threshold: Cosine similarity threshold for clustering.

    Returns:
        List of dicts: [{"canonical": str, "aliases": [str], "score": float}, ...]
    """
    if not terms:
        return []

    if len(terms) == 1:
        return [{
            "canonical": terms[0],
            "aliases": [],
            "score": term_scores.get(terms[0], 0.0),
        }]

    print(f"[Deduplicator] Clustering {len(terms)} terms (threshold={threshold})...")

    # Encode all terms
    embeddings = encode_terms(terms)

    # Normalize for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normalized = embeddings / norms

    # Compute distance matrix (1 - cosine_similarity)
    similarity_matrix = np.dot(normalized, normalized.T)
    distance_matrix = 1.0 - similarity_matrix
    # Ensure no negative distances due to floating point
    distance_matrix = np.maximum(distance_matrix, 0.0)

    # Agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - threshold,  # Convert similarity to distance
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance_matrix)

    # Group terms by cluster
    clusters = {}
    for term, label in zip(terms, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(term)

    # Select canonical form for each cluster
    results = []
    for label, cluster_terms in clusters.items():
        canonical, aliases = _select_canonical(cluster_terms, term_scores)
        best_score = max(term_scores.get(t, 0.0) for t in cluster_terms)
        results.append({
            "canonical": canonical,
            "aliases": aliases,
            "score": best_score,
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Clean up aliases: strip newlines, noise, incomplete parens
    for entry in results:
        cleaned = []
        for alias in entry["aliases"]:
            a = alias.replace("\n", " ").replace("\r", " ")
            a = " ".join(a.split())  # Collapse whitespace
            a = re.sub(r"^(?:said|above|below|mentioned|concerned|further|applicable|net|in case the|actual value of)\s+",
                       "", a, flags=re.IGNORECASE)
            a = re.sub(r"^['\",;:()+\-.\u2013\u2014]+\s*", "", a)
            a = re.sub(r"\s*\([^)]*$", "", a)  # Remove trailing unclosed parens
            a = re.sub(r"\s*['\",;:()+\-.\u2013\u2014]+$", "", a)
            a = re.sub(r"\bTable\s+[IVX]+\b", "", a, flags=re.IGNORECASE)
            a = " ".join(a.split()).strip()
            if a and a.lower() != entry["canonical"].lower() and len(a) > 1:
                cleaned.append(a)
        entry["aliases"] = list(dict.fromkeys(cleaned))  # Dedupe preserving order

    merged_count = len(terms) - len(results)
    print(f"[Deduplicator] {len(terms)} → {len(results)} terms "
          f"({merged_count} merged as duplicates)")

    return results
