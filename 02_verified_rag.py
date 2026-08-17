"""DocuBot - Fase 4: RAG verificado (detector de alucinaciones + fallback).

Flujo: retrieval -> generacion con cita -> 2do call LLM que verifica si la
respuesta esta respaldada por el contexto -> 3 estados posibles.

Uso:  python 02_verified_rag.py "tu pregunta"
"""
from __future__ import annotations

import sys
from enum import Enum
from typing import Literal

from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field

import ingest


class Verdict(str, Enum):
    GROUNDED = "grounded"
    UNSUPPORTED = "unsupported"
    NO_CONTEXT = "no_context"


class JudgeOutput(BaseModel):
    verdict: Literal["grounded", "unsupported", "no_context"] = Field(
        description="grounded si la respuesta esta respaldada por el contexto; "
        "unsupported si la respuesta inventa datos no presentes; "
        "no_context si la pregunta es ajena al corpus."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="confianza del veredicto")
    explanation: str = Field(description="razon breve del veredicto en espanol")


ANSWER_PROMPT = ingest.ANSWER_PROMPT

RELAXED_PROMPT = PromptTemplate(
    """Eres un asistente de IA respondiendo una pregunta.
Puedes usar el contexto si te ayuda, pero tambien puedes usar tu conocimiento
general si lo necesitas. Responde en espanol, de forma breve y directa.

Pregunta: {query_str}
Contexto:
{context_str}

Respuesta:"""
)

JUDGE_PROMPT = PromptTemplate(
    """Eres un verificador de hechos. Tu trabajo es decidir si la RESPUESTA esta
completamente respaldada por el CONTEXTO (fragmentos del corpus).

Reglas:
- grounded: cada afirmacion sustantiva de la respuesta aparece en el contexto.
- unsupported: la respuesta afirma cosas que NO estan en el contexto (inventadas,
  exageradas o de conocimiento general del modelo que el corpus no respalda).
- no_context: la pregunta es ajena al tema del corpus y el contexto no aporta nada
  util para responderla.

Pregunta: {query_str}

CONTEXTO:
{context_str}

RESPUESTA A VERIFICAR:
{answer}

Devuelve el veredicto en formato JSON con los campos: verdict, confidence, explanation.""",
)


NO_INFO = "No tengo informacion sobre eso en el corpus."


def clean_answer(answer: str) -> str:
    """Si la respuesta es el fallback, elimina cualquier cita residual."""
    if answer.strip().startswith(NO_INFO):
        return NO_INFO
    return answer


def verify(answer: str, question: str, context: str) -> JudgeOutput:
    llm = ingest.Settings.llm
    return llm.structured_predict(JudgeOutput, JUDGE_PROMPT, query_str=question,
                                  context_str=context, answer=answer)


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
        print("🤷 NO LO SE: no tengo informacion sobre eso en el corpus")
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