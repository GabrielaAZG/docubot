"""DocuBot - Fase 2/3: RAG minimo con citas.

Uso:  python 01_basic_rag.py "tu pregunta"
"""
from __future__ import annotations

import sys

import ingest

ANSWER_PROMPT = ingest.ANSWER_PROMPT


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python 01_basic_rag.py \"pregunta\"")
        sys.exit(1)
    question = " ".join(sys.argv[1:])

    ingest.configure()
    index = ingest.get_index()
    context, nodes = ingest.retrieve(index, question)

    query_engine = index.as_query_engine(
        text_qa_template=ANSWER_PROMPT,
        similarity_top_k=ingest.SIMILARITY_TOP_K,
    )
    response = query_engine.query(question)

    print("\nPREGUNTA:", question)
    print("\nRESPUESTA:\n", response)
    print("\n--- FUENTES RECUPERADAS ---")
    seen = set()
    for n in nodes:
        title = n.metadata.get("title", "?")
        if title not in seen:
            seen.add(title)
            print(f"- {title}")


if __name__ == "__main__":
    main()