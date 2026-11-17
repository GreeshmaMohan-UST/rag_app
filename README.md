# RAG Machine Learning Knowledge Assistant

Question answering over PDFs with retrieval-augmented generation. Covers document loading and cleaning, chunking,
embeddings with a FAISS vector index, top-k retrieval, grounded answers with citations, and an interactive app.

See [REPORT.md](REPORT.md) for the results, design decisions and limitations.

## Setup

```bash
pip install pypdf numpy faiss-cpu scikit-learn google-genai anthropic
```

Put the PDFs to index in a `data/` folder (not included in this repo).

## Usage

```bash
python build_index.py      # Parts 1-2: load, clean, chunk, embed, build the index
python experiments.py      # Parts 3-4: k and chunk-size experiments, sample answers
python app.py              # Part 5: interactive Q&A (asks for a Gemini key; press Enter to skip)
python app.py --doc NAME   # search only documents whose file name contains NAME
```

Set `RAG_EMBED_BACKEND` to `tfidf` (offline, default), `gemini` or `st` before running `build_index.py`.
Generation uses Gemini when `GEMINI_API_KEY` is set, Claude when `ANTHROPIC_API_KEY` is set, and otherwise an
extractive fallback. API keys are read from environment variables or typed at the prompt; they are never stored in
the repo.
