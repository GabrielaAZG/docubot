"""Descarga el corpus de DocuBot: 10 artículos/ensayos de libre acceso sobre IA.

Cada documento se guarda en docs/ como Markdown limpio con un header de
procedencia (título, autor, URL, licencia/acceso).
"""
from __future__ import annotations

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from html2text import HTML2Text
from pypdf import PdfReader

DOCS = Path(__file__).parent.parent / "docs"
DOCS.mkdir(exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (DocuBot corpus downloader; educational use)"}


def fetch(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return r.text


def html_to_md(html: str) -> str:
    h = HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    return h.handle(html)


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def save(fname: str, header: str, body: str) -> None:
    body = clean_text(body)
    (DOCS / fname).write_text(f"{header}\n\n{body}\n", encoding="utf-8")
    print(f"OK  {fname}  ({len(body)} chars)")


def dl_html(fname: str, url: str, header: str, selector: str | None = None,
            max_chars: int | None = None) -> None:
    html = fetch(url)
    if selector:
        soup = BeautifulSoup(html, "html.parser")
        node = soup.select_one(selector)
        if node is None:
            raise ValueError(f"Selector '{selector}' not found in {url}")
        html = str(node)
    body = html_to_md(html)
    if max_chars and len(body) > max_chars:
        body = body[:max_chars] + "\n\n[El documento se ha truncado para el corpus.]"
    save(fname, header, body)


def dl_pdf(fname: str, url: str, header: str, pages: tuple[int, int] | None = None) -> None:
    r = requests.get(url, headers=UA, timeout=90)
    r.raise_for_status()
    tmp = DOCS / "_tmp.pdf"
    tmp.write_bytes(r.content)
    reader = PdfReader(str(tmp))
    lo, hi = pages or (0, len(reader.pages))
    text = "\n\n".join(p.extract_text() or "" for p in reader.pages[lo:hi])
    tmp.unlink()
    save(fname, header, text)


HEAD = "Título: {title}\nAutor: {author}\nFuente: {url}\nAcceso: {access}"

ARTICLES = [
    dict(
        kind="html", fname="01_rich_sutton_the_bitter_lesson.md",
        url="http://www.incompleteideas.net/IncIdeas/BitterLesson.html",
        header=HEAD.format(title="The Bitter Lesson", author="Richard S. Sutton",
                           url="http://www.incompleteideas.net/IncIdeas/BitterLesson.html",
                           access="Publicación libre del autor"),
    ),
    dict(
        kind="pdf", fname="02_domingos_few_useful_things_ml.md",
        url="https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf",
        header=HEAD.format(title="A Few Useful Things to Know About Machine Learning",
                           author="Pedro Domingos",
                           url="https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf",
                           access="Preprint libre del autor (CACM 2012)"),
        pages=(2, None),
    ),
    dict(
        kind="pdf", fname="03_turing_computing_machinery_intelligence.md",
        url="https://web.archive.org/web/20201226105545id_/https://www.csee.umbc.edu/courses/471/papers/turing.pdf",
        header=HEAD.format(title="Computing Machinery and Intelligence",
                           author="Alan M. Turing",
                           url="https://www.csee.umbc.edu/courses/471/papers/turing.pdf",
                           access="Dominio público (Mind, 1950)"),
        pages=(2, None),
    ),
    dict(
        kind="html", fname="04_vaswani_attention_is_all_you_need.md",
        url="https://arxiv.org/html/1706.03762v7",
        selector="article.ltx_document",
        header=HEAD.format(title="Attention Is All You Need",
                           author="Ashish Vaswani et al.",
                           url="https://arxiv.org/abs/1706.03762",
                           access="arXiv, licencia abierta"),
    ),
    dict(
        kind="html", fname="05_karpathy_unreasonable_effectiveness_rnn.md",
        url="http://karpathy.github.io/2015/05/21/rnn-effectiveness/",
        header=HEAD.format(title="The Unreasonable Effectiveness of Recurrent Neural Networks",
                           author="Andrej Karpathy",
                           url="http://karpathy.github.io/2015/05/21/rnn-effectiveness/",
                           access="Publicación libre del autor"),
        selector="article.post-content",
    ),
    dict(
        kind="html", fname="06_kaplan_scaling_laws_neurips.md",
        url="https://arxiv.org/html/2001.08361",
        selector="article.ltx_document",
        header=HEAD.format(title="Scaling Laws for Neural Language Models",
                           author="Jared Kaplan et al.",
                           url="https://arxiv.org/abs/2001.08361",
                           access="arXiv, licencia abierta"),
    ),
    dict(
        kind="html", fname="07_ji_survey_hallucinations_llms.md",
        url="https://arxiv.org/html/2311.05232v1",
        selector="article.ltx_document",
        header=HEAD.format(title="Survey of Hallucination in Natural Language Generation (LLMs)",
                           author="Ziwei Ji et al.",
                           url="https://arxiv.org/abs/2311.05232",
                           access="arXiv, licencia abierta"),
        max_chars=160000,
    ),
    dict(
        kind="html", fname="08_bereska_mechanistic_interpretability_review.md",
        url="https://arxiv.org/html/2404.14082v3",
        selector="article.ltx_document",
        max_chars=160000,
        header=HEAD.format(title="Mechanistic Interpretability for AI Safety — A Review",
                           author="Leonard Bereska, Efstratios Gavves",
                           url="https://arxiv.org/abs/2404.14082",
                           access="arXiv, licencia abierta"),
    ),
    dict(
        kind="html", fname="09_bengio_machine_learning_composed.md",
        url="https://arxiv.org/html/2112.09332v2",
        selector="article.ltx_document",
        header=HEAD.format(title="Machine Learning for Compositional, and More, Intelligence",
                           author="Yoshua Bengio et al.",
                           url="https://arxiv.org/abs/2112.09332",
                           access="arXiv, licencia abierta"),
        max_chars=120000,
    ),
    dict(
        kind="html", fname="10_wikipedia_retrieval_augmented_generation.md",
        url="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        header=HEAD.format(title="Retrieval-Augmented Generation (artículo de contexto)",
                           author="Wikipedia",
                           url="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
                           access="CC BY-SA 4.0"),
        selector="div.mw-parser-output",
    ),
]


def main() -> None:
    for a in ARTICLES:
        try:
            if a["kind"] == "html":
                dl_html(a["fname"], a["url"], a["header"],
                        a.get("selector"), a.get("max_chars"))
            else:
                dl_pdf(a["fname"], a["url"], a["header"], a.get("pages"))
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {a['fname']}: {e}")


if __name__ == "__main__":
    main()