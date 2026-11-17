"""Part 5: end-to-end interactive RAG Q&A.

Usage:
  python app.py                       interactive session (type a question; /k 7 to change k; /quit to exit)
  python app.py "your question" -k 5  one-shot question

Requires index/ (run build_index.py first). Set ANTHROPIC_API_KEY to use Claude for generation.
"""
import argparse
import sys

from pipeline import answer_question, load_index, prompt_for_gemini_key


def show(result: dict, show_context: bool):
    print("\n" + result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['rank']}] {s['source']}, page {s['page']} (similarity {s['score']})")
    g = result["grounding"]
    print(f"\n[model: {result['model']} | citations: {g['citations']} | lexical support: {g['lexical_support']}"
          f"{' | refused' if g['refused'] else ''}]")
    if show_context:
        for h in result["hits"]:
            print(f"\n--- [{h['rank']}] {h['source']} p{h['page']} ---\n{h['text'][:600]}")


def main():
    parser = argparse.ArgumentParser(description="RAG Q&A over the indexed PDFs")
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    parser.add_argument("-k", type=int, default=5, help="number of chunks to retrieve (default 5)")
    parser.add_argument("--show-context", action="store_true", help="print the retrieved chunk text")
    parser.add_argument("--doc", default=None, help="only search documents whose file name contains this text, "
                                                    "e.g. --doc runbook")
    args = parser.parse_args()

    load_index()  # fail fast if the index is missing
    prompt_for_gemini_key()  # asks only when no key is set and this is a terminal
    if args.question:
        show(answer_question(args.question, k=args.k, source=args.doc), args.show_context)
        return

    print("Ask a question about the indexed documents. Commands: /k N, /doc NAME|all, /context on|off, /quit")
    k, show_ctx, doc = args.k, args.show_context, args.doc
    if doc:
        print(f"Searching only documents matching '{doc}'.")
    while True:
        try:
            q = input("\nQ> ").lstrip("﻿").strip()  # BOM can appear when input is piped from PowerShell
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q.startswith("/k "):
            try:
                k = max(1, int(q.split()[1]))
                print(f"k set to {k}")
            except ValueError:
                print("usage: /k 5")
            continue
        if q.startswith("/context"):
            show_ctx = q.endswith("on")
            print(f"context display {'on' if show_ctx else 'off'}")
            continue
        if q.startswith("/doc"):
            name = q[4:].strip()
            doc = None if name in ("", "all") else name
            print(f"searching: {doc or 'all documents'}")
            continue
        show(answer_question(q, k=k, source=doc), show_ctx)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
