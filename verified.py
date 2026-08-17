"""DocuBot - nucleo de verificacion: 2do call LLM que decide si la respuesta
esta respaldada por el contexto recuperado (grounded/unsupported/no_context).
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field

import ingest

NO_INFO = "No tengo informacion sobre eso en el corpus."

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


def clean_answer(answer: str) -> str:
    """Si la respuesta es el fallback, elimina cualquier cita residual."""
    if answer.strip().startswith(NO_INFO):
        return NO_INFO
    return answer


def verify(answer: str, question: str, context: str) -> JudgeOutput:
    llm = ingest.Settings.llm
    return llm.structured_predict(
        JudgeOutput, JUDGE_PROMPT, query_str=question, context_str=context, answer=answer
    )