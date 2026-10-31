import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag_engine as rag  # noqa: E402


@pytest.fixture(scope="module")
def graph():
    return rag.LegalGraph()


def test_normalize_and_ref():
    assert rag.normalize("¿Artículo 83 TER?") == "articulo 83 ter"
    assert rag.parse_article_ref("qué dice el artículo 83 ter") == ("83", "ter")
    assert rag.parse_article_ref("art. 5") == ("5", "")
    assert rag.parse_article_ref("multas") is None


def test_graph_unique_and_texts(graph):
    assert len(graph.articles) == 465
    assert len({n.id for n in graph.articles}) == 465
    assert all(n.texto for n in graph.articles)


def test_explicit_article(graph):
    r = graph.search("artículo 2 de la Ley de Movilidad")
    assert r.level == 0
    assert r.passages[0].node.articulo == "2"
    assert "Movilidad" in r.passages[0].node.ley


@pytest.mark.parametrize("q", ["me remolcaron la troca", "kiero saber si me pueden multar x velosidad",
                               "xyzzy qwerty"])
def test_never_empty(graph, q):
    r = graph.search(q)
    assert r.passages, q


def test_jerga_expands():
    assert {"arrastre", "grua"} <= set(rag.expand_jerga("me remolcaron la troca"))


def test_expand_hops_grow_and_same_law(graph):
    seed = graph.search("artículo 10 de la Ley de Movilidad", k=1).passages
    h1 = graph.expand(seed, 1)
    h2 = graph.expand(seed, 2)
    assert 1 < len(h1) <= len(h2)
    assert len({p.node.fuente for p in h2}) == 1


def test_build_messages_has_context(graph):
    r = graph.search("artículo 2 de la Ley de Movilidad", k=1)
    msgs = rag.build_messages("¿qué dice?", r.passages, [], False)
    assert msgs[0]["role"] == "system" and "Art. 2" in msgs[-1]["content"]


def test_clean_history():
    h = [{"role": "user", "content": "hola"}, "basura", {"role": "x", "content": "y"},
         {"role": "assistant", "content": ""}] + [{"role": "user", "content": str(i)} for i in range(10)]
    out = rag.clean_history(h)
    assert len(out) == rag.HISTORY_MESSAGES and all(m["role"] == "user" for m in out)
    assert rag.clean_history(None) == []


def _fake_groq(monkeypatch, exc_factory, ok_after=None):
    import groq

    calls = {"n": 0}

    class Completions:
        def create(self, **kw):
            calls["n"] += 1
            if ok_after is not None and calls["n"] > ok_after:
                msg = SimpleNamespace(content="respuesta ok")
                return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
            raise exc_factory(groq)

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(groq, "Groq", lambda **kw: client)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(rag.time, "sleep", lambda s: None)
    return calls


def _rate_limit(groq):
    import httpx

    resp = httpx.Response(429, headers={"retry-after": "0"}, request=httpx.Request("POST", "http://x"))
    return groq.RateLimitError("limit", response=resp, body=None)


def _timeout(groq):
    import httpx

    return groq.APITimeoutError(request=httpx.Request("POST", "http://x"))


def test_429_retries_then_succeeds(monkeypatch):
    calls = _fake_groq(monkeypatch, _rate_limit, ok_after=1)
    assert rag._call_groq([{"role": "user", "content": "hi"}]) == "respuesta ok"
    assert calls["n"] == 2


@pytest.mark.parametrize("factory", [_rate_limit, _timeout])
def test_exhausted_gives_degraded_answer(monkeypatch, factory):
    monkeypatch.setattr(rag, "_engine", None)
    _fake_groq(monkeypatch, factory)
    out = rag.ask_rag("artículo 2 de la Ley de Movilidad")
    assert isinstance(out, str) and "Art. 2" in out and "saturado" in out


def test_ask_rag_never_raises(monkeypatch, tmp_path):
    assert rag.ask_rag("   ") == rag.MSG_EMPTY
    assert rag.ask_rag(None) == rag.MSG_EMPTY
    monkeypatch.setattr(rag, "_engine", None)
    monkeypatch.setattr(rag, "GRAPH_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(rag.LegalGraph.__init__, "__defaults__", (tmp_path / "nope.json",))
    assert rag.ask_rag("hola") == rag.MSG_GENERIC_ERROR


def test_no_api_key_degrades(monkeypatch):
    monkeypatch.setattr(rag, "_engine", None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    out = rag.ask_rag("me remolcaron la troca")
    assert "Art." in out


def test_graph_loaded_once(monkeypatch):
    monkeypatch.setattr(rag, "_engine", None)
    n = {"c": 0}
    orig = rag.LegalGraph.__init__

    def counting(self, *a, **k):
        n["c"] += 1
        orig(self, *a, **k)

    monkeypatch.setattr(rag.LegalGraph, "__init__", counting)
    for _ in range(3):
        rag._get_engine()
    assert n["c"] == 1
