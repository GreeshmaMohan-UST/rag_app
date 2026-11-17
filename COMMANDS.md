# Commands to run the RAG app

Run everything from the `rag_app` folder in a PowerShell or VS Code terminal.

## 1. Install packages (once)

```bash
pip install pypdf numpy faiss-cpu scikit-learn google-genai anthropic
```

## 2. Add the PDFs

Create a `data` folder inside `rag_app` and copy the PDFs into it. The book is not included in the submission.

```text
rag_app/
  data/
    intro-to-ml-book.pdf
    troubleshooting_runbook.pdf
```

## 3. Build the index (Parts 1 and 2)

Option A: offline TF-IDF (default). Needs no API key.

```bash
python build_index.py
```

Option B: Gemini embeddings. Needs a Gemini key, and the free tier takes about 10 minutes.

```powershell
$env:RAG_EMBED_BACKEND = "gemini"
python build_index.py
```

Enter the key when prompted. Press Enter to skip. The key is not saved to disk.

## 4. Run the experiments (Parts 3 and 4)

```bash
python experiments.py
```

This writes the results to `reports/`. Enter a Gemini key when prompted to get generated answers. Press Enter to use the extractive fallback instead.

## 5. Run the app (Part 5)

Ask questions about both documents:

```bash
python app.py
```

Ask questions about the runbook only:

```bash
python app.py --doc runbook
```

Ask one question and exit:

```bash
python app.py "How do I fix an ERR-401 expired token error?" -k 5
```

Commands inside the app:

- `/k 7` changes the number of chunks retrieved
- `/doc runbook` searches only the runbook, and `/doc all` searches both documents
- `/context on` shows the retrieved text, and `/context off` hides it
- `/quit` exits

## Choosing a Gemini model

If Gemini reports that a model is unavailable (404), set a different model name before running:

```powershell
$env:RAG_GEMINI_MODEL = "gemini-3.5-flash-lite"
```

Use the model name the error message suggests. The embedding model can be changed with `RAG_GEMINI_EMBED_MODEL`.

## Troubleshooting

- **429 RESOURCE_EXHAUSTED with "per minute":** wait a minute and run again.
- **429 with "per day":** the daily free quota is used up. Try again tomorrow, or use Option A.
- **503 UNAVAILABLE:** the model is busy. Wait a few minutes, or try another model.
- **Prompt skipped in VS Code:** run the app from the terminal, not with the Run button.
