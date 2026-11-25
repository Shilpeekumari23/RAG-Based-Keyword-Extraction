"""
embedding_ranker.py — Score extracted terms by domain relevance using embeddings.

Uses SentenceTransformers (all-mpnet-base-v2) to compute cosine similarity
between candidate terms and domain seed terms from config.py.
"""

import numpy as np
from typing import List, Dict, Tuple

from config import DOMAIN_SEED_TERMS, EMBEDDING_MODEL, DOMAIN_RELEVANCE_THRESHOLD


# Module-level cache for the embedding model
_model = None
_seed_embeddings = None


def _get_model():
    """Lazy-load the SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  [Embedding] Loading model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  [Embedding] Model loaded successfully")
    return _model


def _get_seed_embeddings():
    """Compute and cache embeddings for domain seed terms."""
    global _seed_embeddings
    if _seed_embeddings is None:
        model = _get_model()
        print(f"  [Embedding] Encoding {len(DOMAIN_SEED_TERMS)} seed terms...")
        _seed_embeddings = model.encode(DOMAIN_SEED_TERMS, show_progress_bar=False)
    return _seed_embeddings


def compute_domain_relevance(terms: List[str]) -> Dict[str, float]:
    """
    Compute domain relevance score for each term based on embedding similarity
    to domain seed terms.

    Score = max cosine similarity to any seed term (0.0 to 1.0).

    Args:
        terms: List of candidate terms.

    Returns:
        Dict mapping term → relevance score.
    """
    if not terms:
        return {}

    model = _get_model()
    seed_embs = _get_seed_embeddings()

    # Encode all candidate terms
    print(f"  [Embedding] Encoding {len(terms)} candidate terms...")
    term_embs = model.encode(terms, show_progress_bar=False)

    # Compute cosine similarity matrix: terms × seeds
    # Normalize for cosine similarity
    term_norms = term_embs / (np.linalg.norm(term_embs, axis=1, keepdims=True) + 1e-10)
    seed_norms = seed_embs / (np.linalg.norm(seed_embs, axis=1, keepdims=True) + 1e-10)

    similarity_matrix = np.dot(term_norms, seed_norms.T)  # shape: (n_terms, n_seeds)

    # Domain relevance = max similarity to any seed term
    max_similarities = np.max(similarity_matrix, axis=1)

    scores = {}
    for term, score in zip(terms, max_similarities):
        scores[term] = float(score)

    return scores


def filter_by_relevance(terms: List[str],
                        threshold: float = DOMAIN_RELEVANCE_THRESHOLD
                        ) -> List[Tuple[str, float]]:
    """
    Filter terms by domain relevance threshold.

    Args:
        terms: List of candidate terms.
        threshold: Minimum relevance score to keep.

    Returns:
        List of (term, score) tuples, sorted by score descending.
    """
    print(f"[Embedding Ranker] Scoring {len(terms)} terms for domain relevance...")

    scores = compute_domain_relevance(terms)

    # Filter and sort
    filtered = [(term, score) for term, score in scores.items() if score >= threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)

    removed = len(terms) - len(filtered)
    print(f"[Embedding Ranker] Kept {len(filtered)} terms "
          f"(removed {removed} below threshold {threshold})")

    return filtered


def encode_terms(terms: List[str]) -> np.ndarray:
    """
    Encode terms into embedding vectors.

    Args:
        terms: List of terms to encode.

    Returns:
        numpy array of shape (n_terms, embedding_dim).
    """
    model = _get_model()
    return model.encode(terms, show_progress_bar=False)
