# RAG-Based Keyword Extraction

Extract ranked technical terms from Indian power-sector PDFs and trace them back to the pages where the pipeline found supporting text. The included [tariff order example](keywords_output.json) contains 80 keywords from a 66-page document. Its first result is **Tariff Order**, classified as **Regulatory** with a ranking score of **0.9143**.

## Run the sample

With Python 3 and Ollama installed, run these commands in a macOS or Linux terminal.

```bash
git clone https://github.com/Shilpeekumari23/RAG-Based-Keyword-Extraction.git
cd RAG-Based-Keyword-Extraction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt openpyxl
python -m spacy download en_core_web_sm
```

Start Ollama locally. If the desktop app already runs the service, continue to the model download. Otherwise, run `ollama serve` in a separate terminal.

```bash
ollama pull qwen2.5:3b
python main.py test.pdf --top-k 20 --output sample_run.json
```

Open `sample_run.json` for the ranked terms or `sample_run.xlsx` for the spreadsheet. The example uses a distinct output name to preserve the checked-in [JSON](keywords_output.json) and [Excel](keywords_output.xlsx) results. Replace `test.pdf` with your PDF path to process another document. The default limit is 80 keywords. Use `--excel` to choose a spreadsheet path and `--verbose` for diagnostic output.

## How extraction works

| Stage | What the code does |
| --- | --- |
| Read and chunk | [PDF loading](pdf_loader.py), [preprocessing](preprocessing.py), and [chunking](chunking.py) turn the document into overlapping text passages with page ranges. |
| Find candidate terms | The [LLM extractor](llm_extractor.py) gives Qwen each passage and a reference vocabulary. The [phrase extractor](noun_phrase_extractor.py) independently proposes terms, and [main.py](main.py) adds exact vocabulary matches from the document. |
| Filter and merge | The [acronym merger](acronym_merger.py) links abbreviations with full forms. The [embedding ranker](embedding_ranker.py) filters candidates for domain relevance, and the [deduplicator](semantic_deduplicator.py) groups similar terms. |
| Rank and export | The [keyword ranker](keyword_ranker.py) selects the top terms. The [output writer](output_writer.py) records scores, categories, and page references derived from chunk ranges. Use these ranges to locate the relevant passage. |

## Model requests and credentials

This is a command-line pipeline. When you run [main.py](main.py), the [LLM extractor](llm_extractor.py) sends a JSON POST to `http://localhost:11434/api/generate`. Each request contains the model name, a prompt with the PDF passage and target vocabulary, and generation settings. The extractor reads Ollama’s response and parses it into candidate terms.

The default local run requires no application login or API key. The request sets only the `Content-Type: application/json` header and sends no bearer token or cookies. Changing `OLLAMA_BASE_URL` in [config.py](config.py) changes which server receives the document passages. The same file sets the Ollama model, embedding model, target vocabulary, and keyword limit.
