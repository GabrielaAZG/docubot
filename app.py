"""DocuBot - UI Streamlit: chat RAG con grounding verificado.

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import ingest
from verified import NO_INFO, RELAXED_PROMPT, clean_answer, verify

st.set_page_config(page_title="DocuBot · RAG verificado", page_icon="📚", layout="wide")

EXTRA_DIR = Path("docs_extra")


@st.cache_resource
def get_index():
    ingest.configure()
    return ingest.get_index()


def ask(question: str, relaxed: bool) -> dict:
    index = get_index()
    context, nodes = ingest.retrieve(index, question)
    query_engine = index.as_query_engine(
        text_qa_template=RELAXED_PROMPT if relaxed else ingest.ANSWER_PROMPT,
        similarity_top_k=ingest.SIMILARITY_TOP_K,
    )
    answer = clean_answer(str(query_engine.query(question)))
    judge = verify(answer, question, context)
    return {
        "answer": answer,
        "verdict": judge.verdict,
        "confidence": judge.confidence,
        "explanation": judge.explanation,
        "sources": sorted({n.metadata.get("title", "?") for n in nodes}),
        "context": context,
        "top_score": getattr(nodes[0], "score", None) if nodes else None,
    }


STYLE = {
    "grounded": ("✅ Respaldada por el corpus", "#e8f5e9", "#2e7d32"),
    "unsupported": ("⚠️ Posible alucinación", "#ffebee", "#c62828"),
    "no_context": ("🤷 No lo sé", "#f5f5f5", "#616161"),
}


def render_message(msg: dict) -> None:
    role, text = msg["role"], msg["content"]
    if role == "user":
        st.markdown(
            f'<div style="background:#e3f2fd;padding:0.7rem 1rem;border-radius:10px;'
            f'margin-bottom:0.5rem;"><b>🧑‍💻 Tú:</b> {text}</div>',
            unsafe_allow_html=True,
        )
        return

    verdict = msg.get("verdict", "grounded")
    label, bg, border = STYLE[verdict]
    header = (
        f'<div style="background:{bg};border-left:5px solid {border};padding:0.8rem 1rem;'
        f'border-radius:10px;margin-bottom:0.5rem;">'
        f'<b style="color:{border};">{label}</b>'
        f'<span style="color:#9e9e9e;font-size:0.85rem;"> · confianza del verificador: '
        f'{msg["confidence"]:.0%}</span>'
    )
    if verdict == "unsupported":
        header += (
            f'<div style="color:{border};font-size:0.9rem;margin-top:0.4rem;">'
            f'La respuesta no está respaldada por el corpus. {msg["explanation"]}</div>'
        )
    elif verdict == "grounded":
        header += (
            f'<div style="color:#616161;font-size:0.9rem;margin-top:0.4rem;">'
            f'{msg["explanation"]}</div>'
        )
    header += f'</div>'
    st.markdown(header, unsafe_allow_html=True)
    st.markdown(msg["content"])
    with st.expander("Fuentes recuperadas del corpus"):
        st.markdown("\n\n---\n\n".join(f"**{s}**" for s in msg["sources"]))
        st.markdown(msg["context"])


def main() -> None:
    st.title("📚 DocuBot — RAG con grounding verificado")
    st.caption(
        "Responde con citas de fuente · detecta alucinaciones · dice "
        "«No lo sé» cuando no hay contexto. "
        "Corpus: 10 artículos/ensayos sobre IA."
    )

    with st.sidebar:
        st.header("Configuración")
        relaxed = st.toggle(
            "Modo demo (alucinación)",
            value=False,
            help="El modelo usa conocimiento general; el detector marca lo que el "
            "corpus no respalda. Para demostrar el estado rojo.",
        )
        st.divider()
        st.subheader("Subir documento extra")
        uploaded = st.file_uploader(
            "Markdown, TXT o PDF", type=["md", "txt", "pdf"], accept_multiple_files=True
        )
        if st.button("Ingerir documentos subidos", disabled=not uploaded):
            EXTRA_DIR.mkdir(exist_ok=True)
            for f in uploaded:
                (EXTRA_DIR / f.name).write_bytes(f.getbuffer())
            with st.spinner("Indexando..."):
                ingest.ingest_extra(EXTRA_DIR)
            get_index.clear()
            st.success(f"{len(uploaded)} documento(s) ingerido(s).")
        if st.button("🗑️ Borrar historial"):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption("Estado del índice: " + ("OK" if (ingest.INDEX_DIR / "chroma.sqlite3").exists() else "vacío"))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        render_message(msg)

    question = st.chat_input("Pregunta sobre los ensayos del corpus...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        render_message({"role": "user", "content": question})
        try:
            with st.spinner("Buscando en el corpus y verificando..."):
                result = ask(question, relaxed)
            bot_msg = {"role": "assistant", "content": result["answer"], **result}
            st.session_state.messages.append(bot_msg)
            render_message(bot_msg)
        except Exception as e:  # noqa: BLE001
            st.error(f"Error al procesar la pregunta: {e}")


if __name__ == "__main__":
    main()