"""Parts 1 and 2: load + explore PDFs, chunk, embed, store in FAISS.

Usage:  python build_index.py
Writes: reports/part1_data_understanding.json, index/ (faiss.index, chunks.json, config.json),
        reports/part2_sample_retrieval.md
"""
import json

from pipeline import (CHUNK_OVERLAP, CHUNK_TOKENS, EMBED_BACKEND, EMBED_DIM, EMBED_MODEL, GEMINI_EMBED_MODEL,
                      INDEX_DIR, REPORT_DIR, build_faiss_index, load_corpus, prompt_for_gemini_key, retrieve,
                      save_index)

SAMPLE_QUERY = "What is the difference between bagging and boosting?"


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    if EMBED_BACKEND == "gemini":
        prompt_for_gemini_key()  # asks for the key if none is set (terminal only)
        embed_desc = f"Gemini {GEMINI_EMBED_MODEL} ({EMBED_DIM}-d; RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions)"
        model_id = GEMINI_EMBED_MODEL
    elif EMBED_BACKEND == "st":
        embed_desc = model_id = EMBED_MODEL
    else:
        embed_desc = model_id = "TF-IDF (1-2 grams, sublinear tf, dense)"
    print(f"Chunking: {CHUNK_TOKENS} tokens, {CHUNK_OVERLAP} overlap; embedding backend {EMBED_BACKEND}: {embed_desc}")
    chunks, doc_stats = load_corpus()
    (REPORT_DIR / "part1_data_understanding.json").write_text(json.dumps(doc_stats, indent=2), encoding="utf-8")
    for s in doc_stats:
        print(json.dumps(s, indent=2))

    print(f"Embedding {len(chunks)} chunks ...")
    index = build_faiss_index(chunks)
    config = {"embed_backend": EMBED_BACKEND,
              "embed_model": model_id,
              "chunk_tokens": CHUNK_TOKENS, "chunk_overlap": CHUNK_OVERLAP,
              "similarity": "cosine (inner product on normalised vectors)", "index_type": "faiss.IndexFlatIP",
              "num_vectors": index.ntotal, "dim": index.d, "documents": [s["document"] for s in doc_stats]}
    save_index(index, chunks, config, INDEX_DIR)
    print("Index saved to", INDEX_DIR)

    hits = retrieve(SAMPLE_QUERY, k=3, index=index, chunks=chunks)
    lines = [f"# Sample retrieval (Part 2)", "", f"Query: `{SAMPLE_QUERY}`", ""]
    for h in hits:
        lines += [f"## Rank {h['rank']} - score {h['score']} - {h['source']}, page {h['page']} - chunk `{h['id']}`", "",
                  "> " + h["text"][:800].replace("\n", "\n> "), ""]
    (REPORT_DIR / "part2_sample_retrieval.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines)[:1500])


if __name__ == "__main__":
    main()
