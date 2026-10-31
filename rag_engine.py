"""Motor GraphRAG sobre las leyes de tránsito y movilidad de Oaxaca.

API pública (única que debe usar el bot)::

    from rag_engine import ask_rag
    respuesta = ask_rag("¿Qué dice el artículo 2?", history=[{"role": "user", "content": "..."}])

Requisitos externos: variable de entorno ``GROQ_API_KEY`` (o un ``.env`` junto a este
módulo). El grafo (``ley_grafo.json``) se localiza relativo a este archivo y se carga una sola
vez por proceso. ``ask_rag`` es síncrona y NUNCA lanza excepciones; si el bot es asíncrono,
llámala con ``await asyncio.to_thread(ask_rag, pregunta, history)``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz, process
from unidecode import unidecode

try:  # python-dotenv es opcional en tiempo de ejecución
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:  # pragma: no cover
    pass

log = logging.getLogger("rag_engine")

BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "ley_grafo.json"
MODEL = os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"

GROQ_TIMEOUT = 15.0
GROQ_MAX_ATTEMPTS = 3
GROQ_TIME_BUDGET = 25.0
MAX_RETRY_WAIT = 8.0

DEFAULT_K = 3
DEFAULT_HOPS = 1
MAX_PASSAGES = 10
MAX_PASSAGE_CHARS = 1800
MAX_CONTEXT_CHARS = 12000
HISTORY_MESSAGES = 6
HISTORY_CHARS = 500
LOW_CONFIDENCE = 0.30

MSG_EMPTY = "Escribe tu pregunta sobre tránsito, vialidad o movilidad en Oaxaca."
MSG_NOT_FOUND = (
    "No encontré un artículo relacionado con tu consulta. Intenta reformularla "
    "o indica la ley o el número de artículo."
)
MSG_GENERIC_ERROR = "Ocurrió un problema al procesar tu consulta. Inténtalo de nuevo en un momento."

# --------------------------------------------------------------------------- texto

STOPWORDS = set(
    """a al algo ante con contra cual cuando de del desde donde e el ella ellos en entre era es esta
    este esto estoy fue ha hay la las le les lo los me mi mis mucho muy ni no nos o para pero por
    que se si sin sobre su sus te tengo ti tu un una unas uno unos y ya yo""".split()
)

# Jerga / coloquialismos -> términos que sí aparecen en la ley (claves normalizadas).
JERGA_A_LEGAL: dict[str, str] = {
    "troca": "camioneta vehiculo",
    "trocas": "camioneta vehiculo",
    "carro": "vehiculo automovil",
    "carros": "vehiculo automovil",
    "coche": "vehiculo automovil",
    "auto": "vehiculo automovil",
    "moto": "motocicleta",
    "camion": "transporte publico vehiculo",
    "combi": "transporte publico",
    "taxi": "transporte publico taxi",
    "remolcaron": "arrastre grua deposito vehiculo remolque",
    "remolcar": "arrastre grua deposito vehiculo remolque",
    "remolque": "arrastre grua deposito vehiculo",
    "grua": "arrastre grua deposito vehiculo",
    "corralon": "deposito vehiculos retiro",
    "arrastraron": "arrastre grua deposito vehiculo",
    "multa": "sancion infraccion multa",
    "multaron": "sancion infraccion multa",
    "infraccion": "sancion infraccion multa",
    "mordida": "cohecho soborno servidor publico",
    "transito": "transito vialidad policia vial",
    "poli": "policia vial agente",
    "tamarindo": "policia vial agente",
    "pico": "alcoholimetro alcohol conductor",
    "borracho": "alcohol aliento etilico conductor",
    "tomado": "alcohol aliento etilico conductor",
    "alcoholimetro": "alcohol aliento etilico conductor",
    "choque": "accidente colision vehiculo",
    "chocaron": "accidente colision vehiculo",
    "accidente": "accidente colision vehiculo",
    "licencia": "licencia conducir",
    "placas": "placas circulacion registro vehiculo",
    "parquimetro": "estacionamiento estacionar via publica",
    "estacionar": "estacionamiento via publica",
    "semaforo": "semaforo senales transito",
    "casco": "casco motociclista",
    "cinturon": "cinturon seguridad",
    "velocidad": "velocidad limite",
}

_ARTICLE_RE = re.compile(
    r"\bart(?:[ií]culos?|\.)?\s*(\d+)\s*(bis|ter|qu[aá]ter|quinquies)?\b", re.IGNORECASE
)


def normalize(text: str) -> str:
    """Minúsculas, sin acentos ni puntuación y con espacios colapsados."""
    text = unidecode(text or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def stem(token: str) -> str:
    """Raíz tosca (prefijo de 5 letras) que tolera plurales, conjugaciones y ruido de OCR."""
    return token[:5] if len(token) > 5 else token


def tokens(text: str) -> list[str]:
    return [t for t in normalize(text).split() if len(t) > 2 and t not in STOPWORDS]


def parse_article_ref(query: str) -> tuple[str, str] | None:
    m = _ARTICLE_RE.search(unidecode(query or ""))
    if not m:
        return None
    suffix = (m.group(2) or "").lower().replace("quater", "quater")
    return m.group(1), suffix


def expand_jerga(query: str) -> list[str]:
    """Tokens de la consulta + equivalentes legales de la jerga detectada."""
    base = tokens(query)
    extra: list[str] = []
    jerga_stems = {stem(k): v for k, v in JERGA_A_LEGAL.items()}
    for tok in base:
        repl = JERGA_A_LEGAL.get(tok) or jerga_stems.get(stem(tok))
        if repl:
            extra.extend(repl.split())
    return base + extra


# --------------------------------------------------------------------------- grafo


@dataclass
class Node:
    id: str
    tipo: str
    articulo: str = ""
    capitulo: str = ""
    ley: str = ""
    fuente: str = ""
    pagina: int = 0
    texto: str = ""


@dataclass
class Passage:
    node: Node
    score: float
    origin: str  # "semilla" | "salto-1" | "salto-2"


@dataclass
class Retrieval:
    passages: list[Passage] = field(default_factory=list)
    level: int = 0
    low_confidence: bool = False
    query_used: str = ""


class LegalGraph:
    """Grafo en memoria: nodos de artículo, adyacencia y un índice léxico invertido."""

    def __init__(self, path: Path = GRAPH_PATH) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.nodes: dict[str, Node] = {}
        for raw in data["nodes"]:
            self.nodes[raw["id"]] = Node(
                id=raw["id"],
                tipo=raw.get("tipo", ""),
                articulo=str(raw.get("articulo", "")),
                capitulo=raw.get("capitulo") or raw.get("nombre", ""),
                ley=raw.get("ley", ""),
                fuente=raw.get("fuente", ""),
                pagina=int(raw.get("pagina") or 0),
                texto=raw.get("texto", ""),
            )
        self.prev: dict[str, str] = {}
        self.next: dict[str, str] = {}
        for e in data["edges"]:
            if e.get("relacion") == "siguiente":
                self.next[e["source"]] = e["target"]
                self.prev[e["target"]] = e["source"]
        self.articles = [n for n in self.nodes.values() if n.tipo == "articulo"]
        self._build_index()

    def _build_index(self) -> None:
        self.docs_norm: list[str] = []
        self.index: dict[str, dict[int, int]] = defaultdict(dict)
        for i, n in enumerate(self.articles):
            norm = normalize(f"{n.ley} {n.capitulo} {n.texto}")
            self.docs_norm.append(norm)
            for t in norm.split():
                if len(t) > 2 and t not in STOPWORDS:
                    s = stem(t)
                    self.index[s][i] = self.index[s].get(i, 0) + 1
        self.vocab = list(self.index)
        n_docs = max(len(self.articles), 1)
        self.idf = {s: math.log(1 + n_docs / (1 + len(d))) for s, d in self.index.items()}

    def neighbors(self, node_id: str) -> list[str]:
        return [x for x in (self.prev.get(node_id), self.next.get(node_id)) if x]

    def expand(self, seeds: list[Passage], hops: int, max_nodes: int = MAX_PASSAGES) -> list[Passage]:
        """BFS de 1–2 saltos por artículos contiguos (misma ley, por construcción)."""
        hops = 2 if hops >= 2 else 1
        out = list(seeds)
        seen = {p.node.id for p in seeds}
        frontier = [p.node.id for p in seeds]
        for depth in range(1, hops + 1):
            nxt: list[str] = []
            for nid in frontier:
                for nb in self.neighbors(nid):
                    if nb not in seen and len(out) < max_nodes:
                        seen.add(nb)
                        out.append(Passage(self.nodes[nb], 0.0, f"salto-{depth}"))
                        nxt.append(nb)
            frontier = nxt
        return out

    # ------------------------------------------------------------ búsqueda difusa
    def _lexical(self, terms: list[str]) -> tuple[list[tuple[int, float]], float]:
        """Puntaje IDF con corrección difusa de términos. Devuelve (ranking, cobertura)."""
        scores: dict[int, float] = defaultdict(float)
        total_mass = matched_mass = 0.0
        for term in dict.fromkeys(stem(t) for t in terms):
            key = term
            if key not in self.index and self.vocab:
                hit = process.extractOne(term, self.vocab, scorer=fuzz.ratio, score_cutoff=80)
                key = hit[0] if hit else None
            weight = self.idf.get(key, 1.5) if key else 1.5
            total_mass += weight
            if not key:
                continue
            matched_mass += weight
            for doc, tf in self.index[key].items():
                scores[doc] += weight * (1 + math.log(tf))
        for doc in scores:
            if "iniciativa" in self.articles[doc].ley.lower():
                scores[doc] *= 0.6  # texto de OCR ruidoso: menor prioridad
        ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranking, (matched_mass / total_mass if total_mass else 0.0)

    def search(self, query: str, k: int = DEFAULT_K, rewritten: str | None = None) -> Retrieval:
        """Recupera semillas por niveles; nunca devuelve una lista vacía si hay artículos."""
        k = max(1, k)
        # Nivel 0: referencia explícita a un artículo
        ref = parse_article_ref(query)
        if ref:
            label = f"{ref[0]} {ref[1]}".strip()
            nq = normalize(query)
            law = None
            if "movilidad" in nq:
                law = "movilidad"
            elif "transito" in nq or "vialidad" in nq:
                law = "transito"
            hits = [n for n in self.articles if n.articulo == label]
            if law:
                filtered = [n for n in hits if law in normalize(n.ley)]
                hits = filtered or hits
            hits.sort(key=lambda n: "iniciativa" in n.ley.lower())
            if hits:
                return Retrieval([Passage(n, 100.0, "semilla") for n in hits[:k]], 0, False, query)

        # Niveles 1-2: normalización + jerga + fuzzy léxico
        terms = expand_jerga(query)
        ranking, coverage = self._lexical(terms)
        level, used = (1 if terms != tokens(query) else 2), query
        # Nivel 3: reescritura externa (opcional) si la cobertura es baja
        if (coverage < LOW_CONFIDENCE or not ranking) and rewritten:
            r2, c2 = self._lexical(expand_jerga(rewritten))
            if r2 and c2 >= coverage:
                ranking, coverage, level, used = r2, c2, 3, rewritten
        # Nivel 4: último recurso, siempre devuelve algo
        if not ranking:
            level = 4
            hits = process.extract(
                normalize(query), self.docs_norm, scorer=fuzz.partial_token_set_ratio, limit=k
            )
            ranking = [(idx, float(score)) for _, score, idx in hits]
        if not ranking:
            return Retrieval([], 4, True, used)
        passages = [Passage(self.articles[i], s, "semilla") for i, s in ranking[:k]]
        return Retrieval(passages, level, coverage < LOW_CONFIDENCE, used)


# --------------------------------------------------------------------------- prompt


SYSTEM_PROMPT = (
    "Eres un asistente jurídico sobre tránsito, vialidad y movilidad en el Estado de Oaxaca. "
    "Responde en español, de forma clara y breve, usando ÚNICAMENTE el contexto legal "
    "proporcionado. Cita siempre la ley y el número de artículo (p. ej. «Ley de Movilidad, "
    "Art. 12»). El texto puede tener errores de OCR: interprétalo con cuidado. Si el contexto no "
    "responde la pregunta, dilo explícitamente y no inventes artículos, montos ni plazos. "
    "No sustituyes asesoría legal profesional."
)


def clean_history(history) -> list[dict]:
    """Conserva los últimos mensajes bien formados, truncados."""
    out: list[dict] = []
    if not isinstance(history, (list, tuple)):
        return out
    for m in history:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                out.append({"role": m["role"], "content": content.strip()[:HISTORY_CHARS]})
    return out[-HISTORY_MESSAGES:]


def format_context(passages: list[Passage], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts: list[str] = []
    used = 0
    for p in passages:
        n = p.node
        head = f"[{n.ley} | Capítulo: {n.capitulo or 's/n'} | Art. {n.articulo} | p. {n.pagina}]"
        block = f"{head}\n{n.texto[:MAX_PASSAGE_CHARS].strip()}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_messages(question: str, passages: list[Passage], history: list[dict], low_conf: bool) -> list[dict]:
    system = SYSTEM_PROMPT
    if low_conf:
        system += (
            " AVISO: la recuperación tuvo baja confianza; si los artículos no son relevantes, "
            "indícalo y sugiere reformular."
        )
    user = f"CONTEXTO LEGAL:\n{format_context(passages)}\n\nPREGUNTA: {question}"
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- Groq


def _call_groq(messages: list[dict], max_tokens: int = 900, deadline: float | None = None) -> str:
    """Llama a Groq con reintentos acotados. Lanza RuntimeError si no es posible."""
    import groq

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY no configurada")
    client = groq.Groq(api_key=key, timeout=GROQ_TIMEOUT, max_retries=0)
    deadline = deadline or time.monotonic() + GROQ_TIME_BUDGET
    last: Exception | None = None
    for attempt in range(GROQ_MAX_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0.0, max_tokens=max_tokens
            )
            return (resp.choices[0].message.content or "").strip()
        except (groq.AuthenticationError, groq.PermissionDeniedError) as e:
            raise RuntimeError(f"credenciales de Groq inválidas: {e}") from e
        except groq.BadRequestError as e:
            raise RuntimeError(f"solicitud rechazada por Groq: {e}") from e
        except (groq.RateLimitError, groq.APITimeoutError, groq.APIConnectionError,
                groq.InternalServerError) as e:
            last = e
            wait = (2 ** attempt) + random.random()
            if isinstance(e, groq.RateLimitError):
                try:
                    wait = float(e.response.headers.get("retry-after", wait))
                except Exception:
                    pass
            wait = min(wait, MAX_RETRY_WAIT)
            log.warning("Groq %s (intento %d/%d)", type(e).__name__, attempt + 1, GROQ_MAX_ATTEMPTS)
            if attempt + 1 >= GROQ_MAX_ATTEMPTS or time.monotonic() + wait > deadline:
                break
            time.sleep(wait)
        except groq.APIStatusError as e:
            raise RuntimeError(f"error de Groq {e.status_code}") from e
    raise RuntimeError(f"Groq no disponible: {type(last).__name__ if last else 'desconocido'}")


def rewrite_query(question: str) -> str | None:
    """Nivel 3: traduce jerga a términos legales. Devuelve None si falla."""
    try:
        out = _call_groq(
            [
                {"role": "system", "content": "Reescribe la consulta del usuario con términos legales "
                 "de tránsito y movilidad de Oaxaca (sin jerga ni faltas). Responde solo con la consulta."},
                {"role": "user", "content": question},
            ],
            max_tokens=60,
            deadline=time.monotonic() + 8,
        )
        return out or None
    except Exception:
        return None


def degraded_answer(passages: list[Passage]) -> str:
    lines = ["El asistente está saturado en este momento; estos son los artículos más relacionados:"]
    for p in passages[:3]:
        n = p.node
        extract = " ".join(n.texto.split())[:400]
        lines.append(f"\n• {n.ley}, Art. {n.articulo} (cap. {n.capitulo or 's/n'}): {extract}…")
    return "\n".join(lines)


# --------------------------------------------------------------------------- motor


class GraphRAGEngine:
    def __init__(self, graph: LegalGraph | None = None) -> None:
        self.graph = graph or LegalGraph()

    def retrieve(self, question: str, history: list[dict] | None = None,
                 k: int = DEFAULT_K, hops: int = DEFAULT_HOPS, use_rewrite: bool = True) -> Retrieval:
        history = clean_history(history)
        query = question
        # Seguimientos cortos ("¿y la multa?"): anteponer la última pregunta del usuario
        if history and len(tokens(question)) <= 3 and not parse_article_ref(question):
            prev = [m["content"] for m in history if m["role"] == "user"]
            if prev:
                query = f"{prev[-1]} {question}"
        r = self.graph.search(query, k)
        if use_rewrite and (r.low_confidence or r.level == 4) and os.getenv("GROQ_API_KEY"):
            rewritten = rewrite_query(question)
            if rewritten:
                r = self.graph.search(query, k, rewritten=rewritten)
        r.passages = self.graph.expand(r.passages, hops)
        return r

    def answer(self, question: str, history: list[dict] | None = None,
               k: int = DEFAULT_K, hops: int = DEFAULT_HOPS) -> str:
        r = self.retrieve(question, history, k, hops)
        if not r.passages or not any(p.node.texto.strip() for p in r.passages):
            return MSG_NOT_FOUND
        messages = build_messages(question, r.passages, clean_history(history), r.low_confidence)
        try:
            return _call_groq(messages)
        except RuntimeError as e:
            log.warning("Modo degradado: %s", e)
            return degraded_answer(r.passages)


_engine: GraphRAGEngine | None = None
_lock = threading.Lock()


def _get_engine() -> GraphRAGEngine:
    """Singleton perezoso: el grafo se carga una sola vez por proceso."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = GraphRAGEngine()
    return _engine


def warmup() -> bool:
    """Carga el grafo por adelantado (opcional). Devuelve False si falla."""
    try:
        _get_engine()
        return True
    except Exception:
        log.exception("No se pudo cargar el grafo")
        return False


def ask_rag(question: str, history: list | None = None) -> str:
    """Responde una pregunta legal. Siempre devuelve ``str`` y nunca lanza excepciones."""
    try:
        if not isinstance(question, str) or not question.strip():
            return MSG_EMPTY
        return _get_engine().answer(question.strip(), history)
    except Exception:
        log.exception("ask_rag falló")
        return MSG_GENERIC_ERROR
