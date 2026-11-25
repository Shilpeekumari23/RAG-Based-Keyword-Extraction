"""
keyword_ranker.py — Final ranking with MMR diversity and category assignment.

Combines domain relevance, document frequency, LLM confidence, and
MMR diversity to produce the final ranked keyword list.
"""

import numpy as np
from typing import List, Dict, Tuple, Set

from config import CATEGORY_TAXONOMY, MMR_LAMBDA, TOP_K_KEYWORDS
from embedding_ranker import encode_terms, compute_domain_relevance


def _compute_document_frequency(term: str, term_lower: str,
                                 chunk_terms: Dict[int, List[str]],
                                 total_chunks: int) -> float:
    """Compute fraction of chunks containing this term."""
    count = 0
    for chunk_id, terms in chunk_terms.items():
        chunk_terms_lower = {t.lower() for t in terms}
        if term_lower in chunk_terms_lower:
            count += 1
    return count / max(total_chunks, 1)


def _assign_category(term: str, term_embedding: np.ndarray,
                     category_embeddings: Dict[str, np.ndarray]) -> str:
    """Assign a category to a term based on embedding similarity."""
    best_category = "Other"
    best_score = -1.0

    term_norm = term_embedding / (np.linalg.norm(term_embedding) + 1e-10)

    for category, cat_embs in category_embeddings.items():
        cat_norms = cat_embs / (np.linalg.norm(cat_embs, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(cat_norms, term_norm)
        max_sim = float(np.max(similarities))

        if max_sim > best_score:
            best_score = max_sim
            best_category = category

    return best_category


def _mmr_rerank(terms: List[str],
                scores: Dict[str, float],
                embeddings: np.ndarray,
                top_k: int,
                lambda_param: float = MMR_LAMBDA) -> List[str]:
    """
    Maximum Marginal Relevance re-ranking.

    Iteratively selects terms that are both relevant and diverse.
    MMR = λ * Relevance - (1-λ) * max_similarity_to_selected

    Args:
        terms: List of candidate terms.
        scores: Dict mapping term → combined relevance score.
        embeddings: Embedding matrix for all terms.
        top_k: Number of terms to select.
        lambda_param: Trade-off parameter (higher = more relevance, less diversity).

    Returns:
        Ordered list of selected terms.
    """
    if len(terms) <= top_k:
        return terms

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normalized = embeddings / norms

    # Similarity matrix
    sim_matrix = np.dot(normalized, normalized.T)

    # Normalize scores to [0, 1]
    score_values = np.array([scores.get(t, 0.0) for t in terms])
    if score_values.max() > score_values.min():
        score_norm = (score_values - score_values.min()) / (score_values.max() - score_values.min())
    else:
        score_norm = np.ones_like(score_values)

    selected_indices = []
    remaining_indices = list(range(len(terms)))

    for _ in range(min(top_k, len(terms))):
        if not remaining_indices:
            break

        best_idx = None
        best_mmr = -float("inf")

        for idx in remaining_indices:
            relevance = score_norm[idx]

            # Max similarity to already selected terms
            if selected_indices:
                max_sim = max(sim_matrix[idx][s] for s in selected_indices)
            else:
                max_sim = 0.0

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [terms[i] for i in selected_indices]


def rank_keywords(deduped_terms: List[Dict],
                  chunk_terms_llm: Dict[int, List[str]],
                  chunk_terms_np: Dict[int, List[str]],
                  total_chunks: int,
                  top_k: int = TOP_K_KEYWORDS) -> List[Dict]:
    """
    Final keyword ranking combining multiple signals.

    Scoring formula:
    - Domain relevance (embedding similarity to seeds): weight 0.4
    - Document frequency (chunk coverage): weight 0.2
    - LLM confidence (extracted by LLM?): weight 0.2
    - MMR diversity (penalize redundancy): weight 0.2

    Args:
        deduped_terms: List from semantic_deduplicator [{canonical, aliases, score}]
        chunk_terms_llm: Dict of chunk_id → LLM-extracted terms
        chunk_terms_np: Dict of chunk_id → NP-extracted terms
        total_chunks: Total number of chunks in the document
        top_k: Number of final keywords to return

    Returns:
        List of ranked keyword dicts with scores and categories.
    """
    print(f"[Keyword Ranker] Ranking {len(deduped_terms)} terms...")

    if not deduped_terms:
        return []

    # Collect all canonical terms
    canonical_terms = [d["canonical"] for d in deduped_terms]
    term_info = {d["canonical"]: d for d in deduped_terms}

    # Merge all chunk terms for frequency computation
    all_chunk_terms = {}
    for cid, terms in chunk_terms_llm.items():
        all_chunk_terms.setdefault(cid, []).extend(terms)
    for cid, terms in chunk_terms_np.items():
        all_chunk_terms.setdefault(cid, []).extend(terms)

    # Build LLM terms set for confidence scoring
    llm_terms_lower = set()
    for terms in chunk_terms_llm.values():
        for t in terms:
            llm_terms_lower.add(t.lower())

    # Compute combined scores
    combined_scores = {}
    for term in canonical_terms:
        term_lower = term.lower()
        info = term_info[term]

        # 1. Domain relevance (from deduplicator, already computed)
        domain_rel = info["score"]

        # 2. Document frequency
        doc_freq = _compute_document_frequency(
            term, term_lower, all_chunk_terms, total_chunks
        )
        # Also check aliases
        for alias in info.get("aliases", []):
            alias_freq = _compute_document_frequency(
                alias, alias.lower(), all_chunk_terms, total_chunks
            )
            doc_freq = max(doc_freq, alias_freq)

        # 3. LLM confidence: was this term or its aliases extracted by the LLM?
        llm_found = term_lower in llm_terms_lower
        if not llm_found:
            for alias in info.get("aliases", []):
                if alias.lower() in llm_terms_lower:
                    llm_found = True
                    break
        llm_conf = 1.0 if llm_found else 0.0

        # Combined score (MMR diversity applied separately)
        score = (0.4 * domain_rel +
                 0.2 * doc_freq +
                 0.2 * llm_conf +
                 0.2 * domain_rel)  # Placeholder for MMR (applied in reranking)

        combined_scores[term] = score

    # Encode terms for MMR and category assignment
    print(f"  [Keyword Ranker] Encoding for MMR and category assignment...")
    term_embeddings = encode_terms(canonical_terms)

    # MMR re-ranking
    ranked_terms = _mmr_rerank(
        canonical_terms, combined_scores, term_embeddings, top_k
    )

    # Compute category embeddings
    print(f"  [Keyword Ranker] Assigning categories...")
    category_embeddings = {}
    for category, exemplars in CATEGORY_TAXONOMY.items():
        category_embeddings[category] = encode_terms(exemplars)

    # Build final results
    results = []
    for rank, term in enumerate(ranked_terms):
        info = term_info[term]
        idx = canonical_terms.index(term)
        term_emb = term_embeddings[idx]

        category = _assign_category(term, term_emb, category_embeddings)

        results.append({
            "term": term,
            "score": round(combined_scores[term], 4),
            "category": category,
            "aliases": info.get("aliases", []),
            "rank": rank + 1,
        })

    print(f"[Keyword Ranker] Final: {len(results)} ranked keywords")
    return results
