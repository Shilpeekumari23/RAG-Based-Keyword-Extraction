"""
llm_extractor.py — Domain-specific keyword extraction using Ollama LLM.

Sends each text chunk to qwen2.5:3b with a carefully crafted prompt
to extract Indian Power Sector terminology.
"""

import json
import re
import time
from typing import List, Dict, Set
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS,
    TARGET_KEYWORDS,
)
from chunking import TextChunk


# Format target keywords list for the prompt context
TARGET_KEYWORDS_STR = "\n".join(f"- {kw}" for kw in TARGET_KEYWORDS)


EXTRACTION_PROMPT = """You are an expert in Indian Power Sector regulation, electricity markets, and energy economics.

From the text below, extract domain-specific technical terms. Specifically, prioritize extracting any of the following standard terms if they are present or referenced in the text:
{target_keywords_list}

RULES:
1. Extract MULTI-WORD technical phrases, not single generic words.
2. DO NOT extract: report, figure, table, chapter, government, policy, analysis, data, results, conclusion, introduction.
3. Include acronyms and their full forms.
4. Extract 15-30 terms per text passage.
5. Return ONLY a JSON array of strings. No explanation.

TEXT:
{text}

EXTRACTED TERMS (JSON array):"""


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama API and return the response text."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
            "top_p": 0.9,
        }
    }).encode("utf-8")

    req = Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "")
    except (URLError, HTTPError) as e:
        raise ConnectionError(f"Ollama API error: {e}")


def _parse_json_array(text: str) -> List[str]:
    """Extract a JSON array from LLM response text, handling malformed output."""
    # Try direct JSON parse first
    text = text.strip()

    # Find JSON array in the response
    # Look for [...] pattern
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return [str(item).strip() for item in arr if isinstance(item, str) and item.strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: extract quoted strings
    terms = re.findall(r'"([^"]{3,})"', text)
    if terms:
        return [t.strip() for t in terms if t.strip()]

    # Last resort: split by newlines and clean
    lines = [l.strip().strip("-•*").strip('"').strip("'").strip()
             for l in text.split("\n") if l.strip()]
    return [l for l in lines if 3 < len(l) < 150 and not l.startswith("{")]


def extract_keywords_from_chunk(chunk: TextChunk, model: str = OLLAMA_MODEL) -> List[str]:
    """
    Extract domain-specific keywords from a single chunk using Ollama LLM.

    Args:
        chunk: TextChunk to extract from.
        model: Ollama model name.

    Returns:
        List of extracted term strings.
    """
    # Truncate chunk if too long (keep within model context)
    text = chunk.text
    words = text.split()
    if len(words) > 1500:
        text = " ".join(words[:1500])

    prompt = EXTRACTION_PROMPT.format(
        target_keywords_list=TARGET_KEYWORDS_STR,
        text=text
    )

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            response = _call_ollama(prompt, model)
            terms = _parse_json_array(response)

            if terms:
                return terms
            else:
                print(f"  [LLM] Chunk {chunk.chunk_id}: Empty response, retrying...")

        except ConnectionError as e:
            print(f"  [LLM] Chunk {chunk.chunk_id}: Connection error: {e}")
            if attempt < LLM_MAX_RETRIES:
                time.sleep(2 ** attempt)

        except Exception as e:
            print(f"  [LLM] Chunk {chunk.chunk_id}: Unexpected error: {e}")
            if attempt < LLM_MAX_RETRIES:
                time.sleep(1)

    print(f"  [LLM] Chunk {chunk.chunk_id}: All retries exhausted, returning empty.")
    return []


def extract_keywords_llm(chunks: List[TextChunk],
                         model: str = OLLAMA_MODEL) -> Dict[int, List[str]]:
    """
    Extract domain-specific keywords from all chunks using the LLM.

    Args:
        chunks: List of TextChunks from the document.
        model: Ollama model name.

    Returns:
        Dict mapping chunk_id → list of extracted terms.
    """
    print(f"[LLM Extractor] Processing {len(chunks)} chunks with {model}...")
    start = time.time()

    results = {}
    all_terms = set()

    for i, chunk in enumerate(chunks):
        print(f"  [LLM] Processing chunk {i + 1}/{len(chunks)} "
              f"(pages {chunk.page_start}-{chunk.page_end})...")
        terms = extract_keywords_from_chunk(chunk, model)
        results[chunk.chunk_id] = terms
        all_terms.update(terms)
        print(f"  [LLM] Chunk {i + 1}: extracted {len(terms)} terms")

    elapsed = time.time() - start
    print(f"[LLM Extractor] Done in {elapsed:.1f}s. "
          f"Total unique terms: {len(all_terms)}")

    return results
