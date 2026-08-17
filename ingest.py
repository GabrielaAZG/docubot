"""DocuBot - modulo compartido: ingesta, indice vectorial y retrieval.

Uso desde 01_basic_rag.py, 02_verified_rag.py y app.py.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"
INDEX_DIR = ROOT / "index"
COLLECTION = "docubot"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
SIMILARITY_TOP_K = 4

_TITLE_RE = re.compile(r"^T[íi]tulo:\s*(.+)$", re.MULTILINE)

ANSWER_PROMPT = PromptTemplate(
    """Eres DocuBot, un asistente especializado en articulos y ensayos sobre IA.
Responde SOLO usando la informacion del contexto proporcionado.
Si el contexto no contiene la respuesta, responde EXACTAMENTE:
"No tengo informacion sobre eso en el corpus." sin texto adicional ni cita.

Al final de tu respuesta, cita la fuente con el formato exacto:
[Fuente: titulo del ensayo]

Pregunta: {query_str}
Contexto:
{context_str}

Respuesta:"""
)


def setup_console() -> None:
    """Consola UTF-8 en Windows (evita mojibake al imprimir respuestas)."""
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def configure() -> None:
    """Configura LLM + embeddings de OpenAI (lee OPENAI_API_KEY del .env)."""
    setup_console()
    load_dotenv(ROOT / ".env")
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )


def _title_from_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = _TITLE_RE.search(text[:500])
    return m.group(1).strip() if m else path.stem


def ingest(docs_dir: Path = DOCS_DIR, persist_dir: Path = INDEX_DIR) -> VectorStoreIndex:
    """Lee los markdown de docs_dir, los trocea, embebe y persiste en ChromaDB."""
    reader = SimpleDirectoryReader(input_dir=str(docs_dir), required_exts=[".md", ".txt"])
    documents = reader.load_data()

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    for doc in documents:
        doc.metadata["title"] = _title_from_file(Path(doc.metadata["file_path"]))

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    print(f"Ingesta completada: {len(documents)} documentos -> {INDEX_DIR.name}/")
    return index


def load_index(persist_dir: Path = INDEX_DIR) -> VectorStoreIndex:
    """Carga el indice persistido sin re-embeder."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store)


def get_index(docs_dir: Path = DOCS_DIR, persist_dir: Path = INDEX_DIR) -> VectorStoreIndex:
    """Reusa el indice persistido; solo ingesta si no existe."""
    if not (persist_dir / "chroma.sqlite3").exists():
        return ingest(docs_dir, persist_dir)
    return load_index(persist_dir)


def retrieve(index: VectorStoreIndex, query: str, top_k: int = SIMILARITY_TOP_K):
    """Retrieval top-k con metadatos y scores. Devuelve (texto_concatenado, nodos)."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    chunks = []
    for n in nodes:
        score = getattr(n, "score", None)
        chunks.append(
            f"[{n.metadata.get('title', 'Sin titulo')}]\n{n.get_content()}"
            + (f"\n(similitud: {score:.3f})" if score is not None else "")
        )
    return "\n\n---\n\n".join(chunks), nodes