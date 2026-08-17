# Test set — DocuBot (10 preguntas)

Conjunto de evaluación manual para demostrar los 3 comportamientos del sistema.
Ejecutar contra la UI (`streamlit run app.py`) o el CLI
(`python 02_verified_rag.py "pregunta"`).

> Nota: las preguntas marcadas **[modo demo]** se ejecutan con `--relaxed`
> (o el toggle "Modo demo" en la UI): el modelo usa conocimiento general y el
> detector marca lo que el corpus no respalda.

## 1. Respuesta correcta en el corpus -> responde con cita (3)

| # | Pregunta | Fuente esperada | Veredicto esperado |
|---|----------|-----------------|--------------------|
| 1 | What is the bitter lesson according to Sutton? | The Bitter Lesson | ✅ grounded |
| 2 | What does the Transformer use instead of recurrence or convolutions? | Attention Is All You Need | ✅ grounded |
| 3 | How should model size and data scale together, according to Kaplan et al.? | Scaling Laws for Neural Language Models | ✅ grounded |

## 2. Respuesta inventable (no está en el corpus) -> detector marca alucinación (2)

| # | Pregunta | Comportamiento esperado |
|---|----------|-------------------------|
| 4 | **[modo demo]** Who invented convolutional neural networks and in what year? | El modelo responde (conocimiento general: LeCun, 1989) y el verificador lo marca ⚠️ unsupported — el corpus no lo menciona |
| 5 | **[modo demo]** How many parameters does GPT-5 have? | El modelo inventa/estima y el verificador lo marca ⚠️ unsupported |

## 3. Preguntas fuera de tema -> "No lo sé" (2)

| # | Pregunta | Comportamiento esperado |
|---|----------|-------------------------|
| 6 | What is the best pizza recipe in Naples? | 🤷 No tengo informacion sobre eso en el corpus |
| 7 | Who won the 2022 World Cup final and what was the score? | 🤷 No tengo informacion sobre eso en el corpus |

## 4. Preguntas de matiz (temas cercanos, retrieval difícil) (3)

| # | Pregunta | Comportamiento esperado |
|---|----------|-------------------------|
| 8 | According to Ji et al., what are the intrinsic and extrinsic types of hallucination? | ✅ grounded si el fragmento correcto se recupera; si se recupera contexto tangencial, el verificador puede marcar ⚠️ — el matiz es el caso difícil |
| 9 | What learning rate schedule did the Transformer use for Adam, and the beta values? | ✅ grounded (el detalle está en el paper, requiere retrieval fino) |
| 10 | According to Domingos, what is the most important lesson about machine learning? | ✅ grounded si recupera el artículo de Domingos; si recupera otro ensayo, ⚠️/🤷 según el verificador |

## Cómo evaluar

Para cada pregunta, comparar el veredicto esperado con el mostrado por el
sistema. El detector no es 100% fiable (ver README, limitaciones conocidas);
una pregunta de matiz puede clasificarse distinto según los fragmentos
recuperados, lo que es parte del comportamiento esperado y documentado.