"""Regenera ley_grafo.json añadiendo el texto de cada artículo.

Lee los documentos ya indexados en chroma_db/chroma.sqlite3 (solo sqlite3, sin
chromadb ni Ollama) y produce un grafo autocontenido para rag_engine.py:
  - cada nodo "articulo" incluye `texto` y `ley`;
  - ids únicos (los duplicados reciben sufijo -2, -3...);
  - capítulos separados por ley (antes `capitulo-i` se compartía entre leyes);
  - la relación `siguiente` solo une artículos de la misma ley.

Uso: python export_texts.py
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SQLITE_PATH = BASE_DIR / "chroma_db" / "chroma.sqlite3"
GRAPH_PATH = BASE_DIR / "ley_grafo.json"
COLLECTION = "ley_transito_oaxaca"


def ley_name(fuente: str) -> str:
    if fuente.startswith("Transito_y_Vialidad_Oaxaca"):
        return "Reglamento de Tránsito Metropolitano para el Municipio de Oaxaca de Juárez"
    if fuente.startswith("Ley_de_Movilidad"):
        return "Ley de Movilidad para el Estado de Oaxaca"
    if fuente.startswith("LEY_DE_INGRESOS"):
        return "Ley de Ingresos del Estado de Oaxaca para el Ejercicio Fiscal 2026"
    if fuente.startswith("Ley_de_Transito"):
        return "Ley de Tránsito y Vialidad del Estado de Oaxaca"
    return "Iniciativa de reforma (9.pdf)"


def alcance(fuente: str) -> str:
    if fuente.startswith("Transito_y_Vialidad_Oaxaca"):
        return "municipal-oaxaca-de-juarez"
    return "estatal-oaxaca"


def ley_slug(fuente: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", ley_name(fuente).lower().replace("á", "a")).strip("-")


def load_units() -> list[dict]:
    con = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
    rows = con.execute(
        """
        SELECT e.id, m.key, m.string_value, m.int_value
        FROM embeddings e
        JOIN segments s ON s.id = e.segment_id
        JOIN collections c ON c.id = s.collection
        JOIN embedding_metadata m ON m.id = e.id
        WHERE c.name = ?
        ORDER BY e.id
        """,
        (COLLECTION,),
    ).fetchall()
    con.close()
    units: dict[int, dict] = {}
    for eid, key, sval, ival in rows:
        value = sval if sval is not None else ival
        units.setdefault(eid, {})["texto" if key == "chroma:document" else key] = value
    return [units[k] for k in sorted(units)]


def build(units: list[dict]) -> dict:
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: dict[str, int] = {}
    chapters: set[str] = set()
    previous: dict[str, str] = {}
    for u in units:
        fuente = u["fuente"]
        slug = ley_slug(fuente)
        chapter_id = f"{slug}:{u['capitulo_id']}"
        if chapter_id not in chapters:
            chapters.add(chapter_id)
            nodes.append(
                {"id": chapter_id, "tipo": "capitulo", "nombre": u.get("capitulo", ""),
                 "ley": ley_name(fuente), "fuente": fuente, "alcance": alcance(fuente)}
            )
        base_id = f"{slug}:{u['capitulo_id']}:articulo-{str(u['articulo']).replace(' ', '-')}"
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        node_id = base_id if seen_ids[base_id] == 1 else f"{base_id}-{seen_ids[base_id]}"
        nodes.append(
            {
                "id": node_id,
                "tipo": "articulo",
                "articulo": str(u["articulo"]),
                "capitulo_id": chapter_id,
                "capitulo": u.get("capitulo", ""),
                "pagina": u.get("pagina", 0),
                "fuente": fuente,
                "ley": ley_name(fuente),
                "alcance": alcance(fuente),
                "texto": u.get("texto", ""),
            }
        )
        edges.append({"source": chapter_id, "target": node_id, "relacion": "contiene"})
        if fuente in previous:
            edges.append({"source": previous[fuente], "target": node_id, "relacion": "siguiente"})
        previous[fuente] = node_id
    return {"nodes": nodes, "edges": edges}


def main() -> None:
    graph = build(load_units())
    GRAPH_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
    arts = sum(1 for n in graph["nodes"] if n["tipo"] == "articulo")
    print(f"Grafo regenerado: {arts} artículos, {len(graph['nodes']) - arts} capítulos, "
          f"{len(graph['edges'])} aristas -> {GRAPH_PATH}")


if __name__ == "__main__":
    main()
