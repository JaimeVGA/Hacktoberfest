import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag_engine as rag  # noqa: E402
import telegram_bot  # noqa: E402


def test_bot_registers_menu_commands_after_startup():
    calls = []

    class FakeBot:
        async def set_my_commands(self, commands):
            calls.extend(commands)

    application = SimpleNamespace(bot=FakeBot())
    asyncio.run(telegram_bot.post_init(application))

    assert [(command.command, command.description) for command in calls] == [
        ("start", "Iniciar o reiniciar la conversación"),
        ("nueva_consulta", "Limpiar contexto para un nuevo caso"),
        ("ayuda", "Guía de uso y cómo redactar tu caso"),
        ("articulos", "Ver normativas indexadas"),
        ("deslinde", "Aviso de exención de responsabilidad legal"),
    ]


@pytest.fixture(scope="module")
def graph():
    return rag.LegalGraph()


def test_normalize_and_ref():
    assert rag.normalize("¿Artículo 83 TER?") == "articulo 83 ter"
    assert rag.parse_article_ref("qué dice el artículo 83 ter") == ("83", "ter")
    assert rag.parse_article_ref("art. 5") == ("5", "")
    assert rag.parse_article_ref("multas") is None


def test_graph_unique_and_texts(graph):
    assert len(graph.articles) > 400
    assert len({n.id for n in graph.articles}) == len(graph.articles)
    assert all(n.texto for n in graph.articles)


def test_explicit_article(graph):
    r = graph.search("artículo 2 de la Ley de Movilidad")
    assert r.level == 0
    assert r.passages[0].node.articulo == "2"
    assert "Movilidad" in r.passages[0].node.ley


def test_city_query_prioritizes_oaxaca_de_juarez(graph):
    r = graph.search("reglamento de tránsito de Oaxaca de Juárez sobre estacionamiento")
    assert r.passages[0].node.alcance == "municipal-oaxaca-de-juarez"


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


def test_municipal_is_default_without_city_mention(graph):
    r = graph.search("multa por estacionarse en doble fila")
    assert r.passages[0].node.alcance == rag.MUNICIPAL


def test_explicit_state_law_not_forced_municipal(graph):
    r = graph.search("artículo 2 de la Ley de Movilidad")
    assert "Movilidad" in r.passages[0].node.ley


def test_article_without_law_prefers_municipal(graph):
    r = graph.search("artículo 5")
    assert r.passages[0].node.alcance == rag.MUNICIPAL


def test_prompt_is_concise_and_municipal():
    p = rag.SYSTEM_PROMPT
    assert "Oaxaca de Juárez" in p and "sin redundancia" in p


def test_disclaimer_added_once_and_stripped_from_history(monkeypatch):
    monkeypatch.setattr(rag, "_engine", None)
    monkeypatch.setattr(rag, "_call_groq", lambda *a, **k: "Respuesta corta (Art. 1).")
    out = rag.ask_rag("multa por estacionarse en doble fila")
    assert out.count(rag.DISCLAIMER) == 1 and out.startswith("Respuesta corta")
    h = rag.clean_history([{"role": "assistant", "content": out}])
    assert rag.DISCLAIMER not in h[0]["content"]


def test_disclaimer_in_degraded_mode(monkeypatch):
    monkeypatch.setattr(rag, "_engine", None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert rag.DISCLAIMER in rag.ask_rag("me remolcaron la troca")


def test_lru_cache_and_sources(monkeypatch):
    monkeypatch.setattr(rag, "_engine", None)
    calls = {"n": 0}

    def fake_call_groq(*a, **k):
        calls["n"] += 1
        return "Respuesta simulada"

    monkeypatch.setattr(rag, "_call_groq", fake_call_groq)
    rag._cache.clear()

    ans1, sources1 = rag.ask_rag_with_sources("¿qué pasa si me remolcan?")
    assert calls["n"] == 1
    assert "Respuesta simulada" in ans1
    assert len(sources1) > 0

    # Segunda llamada idéntica debe salir de caché
    ans2, sources2 = rag.ask_rag_with_sources("¿qué pasa si me remolcan?")
    assert calls["n"] == 1  # No subió el contador de llamadas
    assert ans1 == ans2
    assert sources1 == sources2


def test_get_article_by_id(graph):
    first_art = graph.articles[0]
    node = rag.get_article_by_id(first_art.id)
    assert node is not None
    assert node.id == first_art.id
    assert node.texto != ""
    # También resuelve por short_node_id usado en callback_data de Telegram
    short_id = rag.short_node_id(first_art.id)
    assert rag.get_article_by_id(short_id) == node


def test_sources_keyboard_callback_data_within_telegram_64_byte_limit(graph):
    sources = [(n.id, n.ley, n.articulo) for n in graph.articles]
    markup = telegram_bot.build_sources_keyboard(sources)
    assert markup is not None
    for row in markup.inline_keyboard:
        for btn in row:
            raw_bytes = btn.callback_data.encode("utf-8")
            assert 1 <= len(raw_bytes) <= 64
            assert btn.callback_data.startswith("art:")
            short_id = btn.callback_data[len("art:"):]
            assert rag.get_article_by_id(short_id) is not None


def test_transcribe_audio_mock(monkeypatch):
    class FakeTranscriptions:
        def create(self, **kw):
            return "me multaron por exceso de velocidad"

    class FakeClient:
        audio = SimpleNamespace(transcriptions=FakeTranscriptions())

    monkeypatch.setattr("groq.Groq", lambda **kw: FakeClient())
    monkeypatch.setenv("GROQ_API_KEY", "fake_key")
    result = rag.transcribe_audio(b"dummy_bytes", "audio.ogg")
    assert result == "me multaron por exceso de velocidad"
