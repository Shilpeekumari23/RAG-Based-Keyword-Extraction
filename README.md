# RAG-Based Keyword Extraction for Indian Power-Sector PDFs

## Problem statement

A tariff order can span dozens of pages and use several names for the same concept. A reader looking for technical terms needs to know which phrases matter, how they relate to the power sector, and where to find them in the document. This project turns a PDF into a ranked keyword index with categories, aliases, and page references.

## Approach

The pipeline reads the PDF one passage at a time. A local language model proposes technical terms in context, while phrase extraction and a curated vocabulary supply additional candidates. The pipeline then groups related names and selects terms that cover different topics in the document.

## Methodology and techniques

| Step | Technique | Purpose |
| --- | --- | --- |
| 1. Prepare passages | Clean the extracted text and split it into chunks targeting 1,200 words with a 200-word overlap. | Preserve context across chunk boundaries and retain page ranges for review. |
| 2. Collect candidates | Prompt Qwen with each passage and a power-sector vocabulary. Add spaCy noun phrases, KeyBERT keyphrases, and exact vocabulary matches. | Capture both established terms and phrases suggested by the document itself. |
| 3. Filter and merge | Compare candidate embeddings with domain examples using cosine similarity. Merge acronyms with their full forms and group near-duplicates. | Keep relevant terms and consolidate different names for one concept. |
| 4. Rank and export | Combine domain relevance with extraction signals, then apply Maximal Marginal Relevance (MMR). | Select keywords that are relevant and add variety. Export the results to JSON and Excel. |

The [pipeline entry point](main.py) connects these stages. The [configuration](config.py) defines the vocabulary, chunk sizes, models, and thresholds.

## Results

The included run produced **80 keywords from a 66-page tariff order**. Three entries illustrate the range of topics extracted.

| Extracted term | Assigned category | Ranking score |
| --- | --- | ---: |
| Tariff Order | Regulatory | 0.9143 |
| Benchmark Capital Cost | Economic & Financial | 0.8429 |
| Area Control Error | Grid Operations | 0.8143 |

The scores express the pipeline’s ranking of individual terms. Page references point to the passages that produced each candidate, so a reviewer can check its context in the PDF.

[View the JSON results](keywords_output.json) · [Download the Excel results](keywords_output.xlsx)

## Tools and setup

The project uses Python, Ollama with `qwen2.5:3b`, spaCy, KeyBERT, and Sentence Transformers with `all-mpnet-base-v2`. Excel export uses openpyxl.

### 1. Install the dependencies

With Python 3 and Ollama installed, run the following in a macOS or Linux terminal.

```bash
git clone https://github.com/Shilpeekumari23/RAG-Based-Keyword-Extraction.git
cd RAG-Based-Keyword-Extraction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt openpyxl
python -m spacy download en_core_web_sm
```

### 2. Prepare the local model

Open the Ollama desktop app or run `ollama serve` in a separate terminal. Then download the model.

```bash
ollama pull qwen2.5:3b
```

The [model client](llm_extractor.py) sends each passage and the reference vocabulary to `http://localhost:11434/api/generate`. The default local setup requires no login, API key, or bearer token.

### 3. Run the sample PDF

```bash
python main.py test.pdf --top-k 20 --output sample_run.json
```

Open `sample_run.xlsx` to browse up to 20 keywords, or use `sample_run.json` for further analysis. To process your own document, replace `test.pdf` with its path. Omit `--top-k` to use the default limit of 80.
