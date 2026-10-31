"""CLI interactivo para probar el motor GraphRAG.

Uso:
    python test_cli.py [--hops {1,2}] [-k N] [--no-llm] [--debug]

Comandos dentro del CLI: /ayuda, /hops N, /k N, /debug, /salir
"""

from __future__ import annotations

import argparse

import rag_engine as rag

HELP = "Comandos: /hops 1|2, /k N, /debug, /salir. Escribe tu pregunta legal."


def show_retrieval(r: rag.Retrieval) -> None:
    print(f"\n[recuperación] nivel={r.level} baja_confianza={r.low_confidence} consulta='{r.query_used}'")
    for p in r.passages:
        n = p.node
        print(f"  - {p.origin:8} score={p.score:6.1f} | {n.ley} | cap: {n.capitulo or 's/n'} | "
              f"Art. {n.articulo} | p.{n.pagina}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CLI de prueba del GraphRAG legal")
    ap.add_argument("--hops", type=int, choices=(1, 2), default=rag.DEFAULT_HOPS)
    ap.add_argument("-k", type=int, default=rag.DEFAULT_K)
    ap.add_argument("--no-llm", action="store_true", help="solo mostrar la recuperación")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    hops, k, debug = args.hops, args.k, args.debug

    print(f"GraphRAG legal ({rag.MODEL}). {HELP}")
    history: list[dict] = []
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            if cmd in ("/salir", "/exit", "/q"):
                break
            if cmd == "/hops" and arg.strip() in ("1", "2"):
                hops = int(arg)
            elif cmd == "/k" and arg.strip().isdigit() and int(arg) > 0:
                k = int(arg)
            elif cmd == "/debug":
                debug = not debug
                print(f"debug={'on' if debug else 'off'}")
            else:
                print(HELP)
            continue
        if debug or args.no_llm:
            show_retrieval(rag._get_engine().retrieve(line, history, k, hops, use_rewrite=not args.no_llm))
        if args.no_llm:
            continue
        # misma ruta que usará el bot (usa los valores por defecto de k y hops salvo cambios)
        answer = rag._get_engine().answer(line, history, k, hops) if (k, hops) != (
            rag.DEFAULT_K, rag.DEFAULT_HOPS) else rag.ask_rag(line, history)
        print(f"\n{answer}")
        history += [{"role": "user", "content": line}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
