"""Parts 3 and 4: experiments on k and chunk size, plus grounded answer samples.

Usage:  python experiments.py            (set ANTHROPIC_API_KEY to generate with Claude; otherwise extractive fallback)
Writes: reports/part3_k_experiment.json, reports/part3_chunking_experiment.json,
        reports/part4_sample_answers.md

Retrieval quality is measured with a small hand-labelled set: a query counts as a "hit" at k when at least one
of the top-k chunks contains one of the expected keywords. This is a transparent proxy, not a full relevance judgement.
"""
import json
import re

from pipeline import (CHUNK_OVERLAP, CHUNK_TOKENS, EMBED_BACKEND, REPORT_DIR, answer_question, build_faiss_index,
                      load_corpus, load_index, prompt_for_gemini_key, retrieve)

EVAL_SET = [
    # Hands-On ML style book (intro-to-ml-book.pdf)
    {"q": "What is overfitting and how can it be detected?", "any": ["overfit"]},
    {"q": "What is the difference between bagging and boosting?", "any": ["bagging", "boosting"]},
    {"q": "How does gradient descent update the model parameters?", "any": ["gradient descent"]},
    {"q": "What is regularization used for?", "any": ["regulariz"]},
    {"q": "What is cross-validation?", "any": ["cross-validation", "cross validation"]},
    {"q": "What do precision and recall measure?", "any": ["precision"]},
    # Troubleshooting runbook (troubleshooting_runbook.pdf)
    {"q": "How do I fix an ERR-401 expired token error?", "any": ["ERR-401", "token_expired"]},
    {"q": "What causes connection pool exhaustion (DB-POOL)?", "any": ["DB-POOL", "connection leak"]},
    {"q": "What should I do when a container is OOMKilled?", "any": ["OOMKilled"]},
    {"q": "How do I renew an expired TLS certificate?", "any": ["cert-manager", "CERT-EXP"]},
    {"q": "What backoff should clients use for ERR-429 rate limits?", "any": ["backoff", "ERR-429"]},
    {"q": "What is the acknowledgement target for a Sev1 incident?", "any": ["Sev1"]},
]
K_VALUES = [3, 5, 7]
OUT_OF_SCOPE = "What is the best recipe for sourdough bread?"


def is_hit(hits, keywords):
    text = " ".join(h["text"] for h in hits).lower()
    return any(k.lower() in text for k in keywords)


def run_k_experiment(index, chunks):
    results = {}
    for k in K_VALUES:
        hits_per_q = []
        for item in EVAL_SET:
            hits = retrieve(item["q"], k=k, index=index, chunks=chunks)
            hits_per_q.append({"query": item["q"], "hit": is_hit(hits, item["any"]),
                               "top_sources": [f"{h['source']} p{h['page']}" for h in hits]})
        hit_rate = sum(r["hit"] for r in hits_per_q) / len(hits_per_q)
        results[f"k={k}"] = {"hit_rate": round(hit_rate, 3), "context_tokens_approx": k * CHUNK_TOKENS,
                             "per_query": hits_per_q}
        print(f"k={k}: hit rate {hit_rate:.2%}")
    return results


def run_chunking_experiment():
    """Rebuild the index for several chunk sizes and compare hit rate at k=5."""
    configs = [(250, 50), (500, 100), (1000, 150)]
    summary = {}
    for size, overlap in configs:
        chunks, doc_stats = load_corpus(size=size, overlap=overlap)
        index = build_faiss_index(chunks)
        hits_ok = sum(is_hit(retrieve(e["q"], 5, index, chunks), e["any"]) for e in EVAL_SET)
        summary[f"{size}/{overlap}"] = {
            "num_chunks": len(chunks),
            "hit_rate_at_5": round(hits_ok / len(EVAL_SET), 3),
            "chunk_tokens": size, "overlap_tokens": overlap,
            "per_document_chunks": {s["document"]: s["chunking"]["num_chunks"] for s in doc_stats},
        }
        print(f"chunk {size}/{overlap}: {len(chunks)} chunks, hit@5 {hits_ok}/{len(EVAL_SET)}")
    return summary


def write_sample_answers(index, chunks):
    queries = [e["q"] for e in EVAL_SET[::3]] + [OUT_OF_SCOPE]
    lines = ["# Sample grounded answers (Part 4)", "", f"k = 5, chunk size = {CHUNK_TOKENS}, overlap = {CHUNK_OVERLAP}", ""]
    records = []
    for q in queries:
        res = answer_question(q, k=5)
        records.append({k: v for k, v in res.items() if k != "hits"})
        lines += [f"## Q: {q}", "", f"**Model:** {res['model']}", "", res["answer"], "",
                  f"**Sources:** " + "; ".join(f"[{s['rank']}] {s['source']} p{s['page']} (score {s['score']})"
                                               for s in res["sources"]),
                  f"**Grounding:** `{json.dumps(res['grounding'])}`", ""]
    (REPORT_DIR / "part4_sample_answers.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / "part4_sample_answers.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("\n".join(lines)[:3000])


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    prompt_for_gemini_key()  # asks only when no key is set and this is a terminal
    index, chunks = load_index()
    k_results = run_k_experiment(index, chunks)
    (REPORT_DIR / "part3_k_experiment.json").write_text(json.dumps(k_results, indent=2), encoding="utf-8")
    write_sample_answers(index, chunks)
    # Last: this rebuilds indexes in memory and refits the vectoriser, so it must not run before the steps above.
    # With Gemini embeddings it would re-embed ~900 chunks, which exceeds the free-tier quota, so it runs only for
    # the offline backends. The TF-IDF results are kept in reports_tfidf/.
    if EMBED_BACKEND == "gemini":
        print("Skipping the chunk-size experiment for the gemini backend (free-tier quota). See reports_tfidf/.")
        return
    chunk_results = run_chunking_experiment()
    (REPORT_DIR / "part3_chunking_experiment.json").write_text(json.dumps(chunk_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
