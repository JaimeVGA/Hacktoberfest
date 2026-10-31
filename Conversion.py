"""Indexa la ley en RAG por capítulo y artículo.

Uso:
    python Conversion.py ingest
    python Conversion.py query "¿Qué dice el artículo 2?"

Requiere Ollama activo y, por defecto:
    ollama pull nomic-embed-text
    ollama pull qwen2.5:1.5b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Ley_de_Transito_y_vialidad_(Ref_dto_752_aprob_LXVI_Legis_2_sep_2025_PO_38_8a_secc_20_sep_2025).pdf"
DB_DIR = BASE_DIR / "chroma_db"
GRAPH_PATH = BASE_DIR / "ley_grafo.json"

EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:1.5b")
OCR_VERSION = "v2"
OCR_DPI = os.getenv("OCR_DPI", "300")
OCR_PSM = os.getenv("OCR_PSM", "3")
LOCAL_TESSDATA = BASE_DIR / "tessdata"

CHAPTER_RE = re.compile(
    r"(?m)^\s*CAP[IÍ]TULO\s+([IVXLCDM]+|[0-9]+)\s*$"
    r"(?:\s*\n\s*(.+))?$",
    re.IGNORECASE,
)
ARTICLE_RE = re.compile(
    r"(?m)^\s*Art[ií]culo\s+(\d+)"
    r"(?:\s*(Bis|Ter|Qu[aá]ter|Qu[aá]te\s*r|Quinquies))?"
    r"(?:\s*[-.]\s*|\s+)",
    re.IGNORECASE,
)


def _clean_page(text: str) -> str:
    """Quita encabezados y pies repetidos del periódico oficial."""
    lines = []
    for line in text.splitlines():
        normalized = " ".join(line.split()).lower()
        if not normalized or normalized.startswith("decreto 1801"):
            continue
        if "h. congreso del estado libre" in normalized:
            continue
        if "lxvi legislatura constitucional" in normalized:
            continue
        if "dirección de informática" in normalized:
            continue
        lines.append(line.rstrip())
    return "\n".join(lines)


def _chapter_at(text: str, position: int) -> tuple[str, str]:
    matches = list(CHAPTER_RE.finditer(text, 0, position))
    if not matches:
        return "preámbulo", ""
    match = matches[-1]
    name = " ".join((match.group(2) or "").split())
    return f"capitulo-{match.group(1).lower()}", name


def _ocr_pdf(pdf_path: Path) -> list[Document]:
    """Convierte temporalmente cada página escaneada a texto con Tesseract."""
    pages: list[Document] = []
    languages = os.getenv("OCR_LANG", "spa+eng")
    tessdata_prefix = str(LOCAL_TESSDATA) if (LOCAL_TESSDATA / "spa.traineddata").exists() else None
    if tessdata_prefix is None:
        languages = "eng"
        print("Advertencia: falta tessdata/spa.traineddata; usar OCR inglés.")
    with tempfile.TemporaryDirectory(prefix="rag-ocr-") as temp_dir:
        prefix = str(Path(temp_dir) / "page")
        rendered = subprocess.run(
            ["pdftoppm", "-r", OCR_DPI, "-gray", "-png", str(pdf_path), prefix],
            check=True,
            capture_output=True,
            text=True,
        )
        del rendered
        for image_path in sorted(Path(temp_dir).glob("page-*.png")):
            enhanced_path = image_path.with_name(f"{image_path.stem}-enhanced.png")
            subprocess.run(
                [
                    "magick", str(image_path), "-deskew", "40%",
                    "-contrast-stretch", "0x10%", "-sharpen", "0x1",
                    str(enhanced_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [
                    "tesseract", str(enhanced_path), "stdout",
                    "-l", languages, "--psm", OCR_PSM,
                ],
                env={**os.environ, **({"TESSDATA_PREFIX": tessdata_prefix} if tessdata_prefix else {})},
                check=True,
                capture_output=True,
                text=True,
            )
            pages.append(Document(page_content=result.stdout, metadata={"page": len(pages) + 1}))
    cache = pdf_path.with_suffix(f".ocr.{OCR_VERSION}.txt")
    cache.write_text("\n\n".join(page.page_content for page in pages), encoding="utf-8")
    return pages


def _load_pages(pdf_path: Path) -> list[Document]:
    pages = PyPDFLoader(str(pdf_path)).load()
    if sum(len(page.page_content.strip()) for page in pages) < 1000:
        cache = pdf_path.with_suffix(f".ocr.{OCR_VERSION}.txt")
        if cache.exists():
            return [
                Document(page_content=text, metadata={"page": index + 1})
                for index, text in enumerate(cache.read_text(encoding="utf-8").split("\n\n"))
            ]
        print(f"OCR: {pdf_path.name} ({len(pages)} páginas)")
        pages = _ocr_pdf(pdf_path)
    return pages


def extract_legal_units(pdf_path: Path = PDF_PATH) -> list[Document]:
    """Extrae un documento por artículo, con capítulo y página de origen."""
    pages = _load_pages(pdf_path)
    cleaned_pages = [_clean_page(page.page_content) for page in pages]
    full_text = "\n".join(cleaned_pages)
    page_ends = []
    offset = 0
    for page_text in cleaned_pages:
        offset += len(page_text) + 1
        page_ends.append(offset)
    articles = list(ARTICLE_RE.finditer(full_text))
    if not articles:
        raise ValueError("No se encontraron artículos en el PDF.")

    units: list[Document] = []
    for index, match in enumerate(articles):
        start = match.start()
        end = articles[index + 1].start() if index + 1 < len(articles) else len(full_text)
        content = full_text[start:end].strip()
        chapter_id, chapter_name = _chapter_at(full_text, start)
        article_number = match.group(1)
        suffix = re.sub(r"\s+", "", (match.group(2) or "").lower())
        article_label = f"{article_number} {suffix}".strip()
        page = next((i + 1 for i, end in enumerate(page_ends) if start < end), None)
        units.append(
            Document(
                page_content=content,
                metadata={
                    "tipo": "articulo",
                    "articulo": article_label,
                    "capitulo_id": chapter_id,
                    "capitulo": chapter_name,
                    "pagina": page or 0,
                    "fuente": pdf_path.name,
                    "unidad_id": (
                        f"{pdf_path.stem[:24]}-{chapter_id}-"
                        f"articulo-{article_label.replace(' ', '-')}"
                    ),
                },
            )
        )
    return units


def build_graph(units: Iterable[Document], output: Path = GRAPH_PATH) -> None:
    """Guarda un grafo pequeño y portable para expandir el contexto recuperado."""
    nodes: list[dict] = []
    edges: list[dict] = []
    previous_article: str | None = None
    chapters: set[str] = set()
    for unit in units:
        meta = unit.metadata
        article_id = meta["unidad_id"]
        chapter_id = meta["capitulo_id"]
        if chapter_id not in chapters:
            chapters.add(chapter_id)
            nodes.append({"id": chapter_id, "tipo": "capitulo", "nombre": meta["capitulo"]})
        nodes.append({"id": article_id, "tipo": "articulo", **meta})
        edges.append({"source": chapter_id, "target": article_id, "relacion": "contiene"})
        if previous_article:
            edges.append({"source": previous_article, "target": article_id, "relacion": "siguiente"})
        previous_article = article_id
    output.write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest() -> None:
    pdfs = sorted(BASE_DIR.glob("*.pdf"))
    units: list[Document] = []
    for pdf_path in pdfs:
        try:
            units.extend(extract_legal_units(pdf_path))
        except ValueError as error:
            print(f"Omitido {pdf_path.name}: {error}")
    if not units:
        raise RuntimeError("No se pudieron extraer artículos de los PDFs.")
    build_graph(units)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name="ley_transito_oaxaca",
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )
    vectorstore.delete_collection()
    Chroma.from_documents(
        documents=units,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name="ley_transito_oaxaca",
    )
    print(f"Indexados {len(units)} artículos de {len(pdfs)} PDFs en {DB_DIR}")
    print(f"Grafo guardado en {GRAPH_PATH}")


def query(question: str) -> None:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name="ley_transito_oaxaca",
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM(model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Responde en español usando solo el contexto legal. "
                "Cita capítulo y artículo. Si no está en el contexto, indica que no consta.\n\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    documents = retriever.invoke(question)
    context = "\n\n".join(
        f"[{doc.metadata['capitulo']} / Artículo {doc.metadata['articulo']}]\n{doc.page_content}"
        for doc in documents
    )
    answer = (prompt | llm).invoke({"context": context, "question": question})
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG jurídico de la Ley de Tránsito de Oaxaca")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="crear Chroma y el grafo")
    query_parser = subparsers.add_parser("query", help="consultar el índice")
    query_parser.add_argument("question")
    args = parser.parse_args()
    if args.command == "ingest":
        ingest()
    else:
        query(args.question)


if __name__ == "__main__":
    main()
