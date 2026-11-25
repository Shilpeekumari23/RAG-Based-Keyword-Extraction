"""
acronym_merger.py — Detect, link, and merge acronyms with their expanded forms.

Builds a bidirectional map between acronyms and expansions found in the document,
supplemented by a curated dictionary from config.py.
"""

import re
from typing import Dict, List, Tuple, Set

from config import KNOWN_ACRONYMS


class AcronymMap:
    """Bidirectional map between acronyms and their expansions."""

    def __init__(self):
        # acronym → expansion (canonical)
        self.acronym_to_expansion: Dict[str, str] = {}
        # expansion (lowercase) → acronym
        self.expansion_to_acronym: Dict[str, str] = {}

    def add(self, acronym: str, expansion: str):
        """Add an acronym-expansion pair."""
        acronym = acronym.strip().upper()
        expansion = expansion.strip()

        if not acronym or not expansion:
            return

        # Store mappings
        self.acronym_to_expansion[acronym] = expansion
        self.expansion_to_acronym[expansion.lower()] = acronym

    def get_expansion(self, acronym: str) -> str:
        """Get expansion for an acronym, or return the acronym itself."""
        return self.acronym_to_expansion.get(acronym.upper(), acronym)

    def get_acronym(self, expansion: str) -> str:
        """Get acronym for an expansion, or return empty string."""
        return self.expansion_to_acronym.get(expansion.lower(), "")

    def get_canonical(self, term: str) -> Tuple[str, List[str]]:
        """
        Get the canonical form and aliases for a term.

        Returns: (canonical_term, [aliases])
        - If term is an acronym → canonical is expansion, alias is acronym
        - If term is an expansion → canonical is expansion, alias is acronym
        - Otherwise → canonical is the term itself, no aliases
        """
        term_upper = term.strip().upper()
        term_lower = term.strip().lower()

        # Check if it's a known acronym
        if term_upper in self.acronym_to_expansion:
            expansion = self.acronym_to_expansion[term_upper]
            return expansion, [term_upper]

        # Check if it's a known expansion
        if term_lower in self.expansion_to_acronym:
            acronym = self.expansion_to_acronym[term_lower]
            return term.strip(), [acronym]

        # Check partial match (e.g., "Renewable Purchase Obligations" → "Renewable Purchase Obligation")
        for exp_lower, acr in self.expansion_to_acronym.items():
            if (term_lower.rstrip("s") == exp_lower.rstrip("s") or
                exp_lower.rstrip("s") == term_lower.rstrip("s")):
                return self.acronym_to_expansion.get(acr, term.strip()), [acr]

        return term.strip(), []

    def __len__(self):
        return len(self.acronym_to_expansion)

    def __repr__(self):
        return f"AcronymMap({len(self)} entries)"


def _extract_acronyms_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Extract acronym-expansion pairs from document text.

    Patterns detected:
    - "Full Expansion (ACRONYM)"
    - "ACRONYM (Full Expansion)"
    - "Full Expansion – ACRONYM"
    """
    pairs = []

    # Pattern 1: "Expansion (ACRONYM)"
    for match in re.finditer(
        r"([A-Z][a-zA-Z]+(?:\s+(?:and|&|of|for|the|in|on|to|with)?\s*[a-zA-Z]+){1,8})\s*\(([A-Z][A-Z&]{1,9})\)",
        text
    ):
        expansion = match.group(1).strip()
        acronym = match.group(2).strip()
        # Validate: acronym should roughly match initials
        if len(acronym) >= 2 and len(expansion.split()) >= 2:
            pairs.append((acronym, expansion))

    # Pattern 2: "ACRONYM (Expansion)"
    for match in re.finditer(
        r"\b([A-Z][A-Z&]{1,9})\s*\(([A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+){1,8})\)",
        text
    ):
        acronym = match.group(1).strip()
        expansion = match.group(2).strip()
        if len(acronym) >= 2 and len(expansion.split()) >= 2:
            pairs.append((acronym, expansion))

    # Pattern 3: "Expansion – ACRONYM" or "Expansion - ACRONYM"
    for match in re.finditer(
        r"([A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+){1,8})\s*[-–—]\s*([A-Z]{2,8})\b",
        text
    ):
        expansion = match.group(1).strip()
        acronym = match.group(2).strip()
        if len(acronym) >= 2:
            pairs.append((acronym, expansion))

    return pairs


def build_acronym_map(full_text: str) -> AcronymMap:
    """
    Build a comprehensive acronym map from document text and known dictionary.

    Args:
        full_text: The full document text.

    Returns:
        AcronymMap with all detected and known acronym-expansion pairs.
    """
    acr_map = AcronymMap()

    # 1. Load known acronyms from config
    for acronym, expansion in KNOWN_ACRONYMS.items():
        acr_map.add(acronym, expansion)

    # 2. Extract acronyms from document text
    doc_pairs = _extract_acronyms_from_text(full_text)
    for acronym, expansion in doc_pairs:
        # Document-specific definitions override defaults
        acr_map.add(acronym, expansion)

    print(f"[Acronym Merger] Built map with {len(acr_map)} entries "
          f"({len(KNOWN_ACRONYMS)} known + {len(doc_pairs)} from document)")

    return acr_map


def merge_terms_with_acronyms(terms: List[str], acr_map: AcronymMap) -> List[Dict]:
    """
    Process a list of terms, merging acronyms with their expansions.

    Args:
        terms: List of raw extracted terms.
        acr_map: AcronymMap built from the document.

    Returns:
        List of dicts: [{"canonical": str, "aliases": [str], "original": str}, ...]
    """
    results = []
    seen_canonical = {}  # canonical_lower → index in results

    for term in terms:
        canonical, aliases = acr_map.get_canonical(term)
        canonical_lower = canonical.lower()

        if canonical_lower in seen_canonical:
            # Merge aliases into existing entry
            idx = seen_canonical[canonical_lower]
            existing_aliases = set(results[idx]["aliases"])
            existing_aliases.update(aliases)
            if term.strip() != canonical:
                existing_aliases.add(term.strip())
            results[idx]["aliases"] = list(existing_aliases)
        else:
            # New canonical term
            all_aliases = set(aliases)
            if term.strip().lower() != canonical_lower:
                all_aliases.add(term.strip())
            results.append({
                "canonical": canonical,
                "aliases": list(all_aliases),
                "original": term.strip(),
            })
            seen_canonical[canonical_lower] = len(results) - 1

    print(f"[Acronym Merger] Merged {len(terms)} terms → {len(results)} canonical terms")
    return results
