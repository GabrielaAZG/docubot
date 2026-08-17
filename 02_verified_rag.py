"""DocuBot - Fase 4 (CLI): RAG verificado (detector de alucinaciones + fallback).

Flujo: retrieval -> generacion con cita -> 2do call LLM que verifica si la
respuesta esta respaldada por el contexto -> 3 estados posibles.

Uso:
  python 02_verified_rag.py "tu pregunta"
  python 02_verified_rag.py --relaxed "tu pregunta"   # modo demo
"""
from __future__ import annotations

import sys

from verified import NO_INFO, RELAXED_PROMPT, Verdict, clean_answer, verify

import ingest


def main() -> None:
    args = sys.argv[1:]
    relaxed = "--relaxed" in args
    args = [a for a in args if a != "--relaxed"]
    if not args:
        print("Uso: python 02_verified_rag.py [--relaxed] \"pregunta\"")
        print("  --relaxed: modo demo - el modelo usa conocimiento general y el")
        print("  detector de alucinaciones marca lo que el corpus no respalda.")
        sys.exit(1)
    question = " ".join(args)

    ingest.configure()
    index = ingest.get_index()
    context, nodes = ingest.retrieve(index, question)

    top_score = getattr(nodes[0], "score", None) if nodes else None
    query_engine = index.as_query_engine(
        text_qa_template=RELAXED_PROMPT if relaxed else ingest.ANSWER_PROMPT,
        similarity_top_k=ingest.SIMILARITY_TOP_K,
    )
    answer = clean_answer(str(query_engine.query(question)))
    judge = verify(answer, question, context)

    print("\nPREGUNTA:", question)
    print("\nRESPUESTA DEL MODELO:\n", answer)

    print("\n" + "=" * 60)
    if judge.verdict == Verdict.GROUNDED.value:
        print("✅ RESPALDADA POR EL CORPUS (grounded)")
        print(f"   Confianza del verificador: {judge.confidence:.2f}")
    elif judge.verdict == Verdict.UNSUPPORTED.value:
        print("⚠️ POSIBLE ALUCINACION (respuesta no respaldada por el corpus)")
        print(f"   Confianza del verificador: {judge.confidence:.2f}")
        print("\n--- Lo que dice el corpus sobre el tema ---")
        print(context[:3000])
    else:
        print(f"🤷 {NO_INFO}")
        print(f"   Confianza del verificador: {judge.confidence:.2f}")
    print(f"\nExplicacion del verificador: {judge.explanation}")
    if top_score is not None:
        print(f"\nMejor similitud del retrieval: {top_score:.3f}")
    print("=" * 60)

    print("\nFUENTES RECUPERADAS:")
    seen = set()
    for n in nodes:
        title = n.metadata.get("title", "?")
        if title not in seen:
            seen.add(title)
            print(f"- {title}")


if __name__ == "__main__":
    main()