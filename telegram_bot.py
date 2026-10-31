"""Bot de Telegram para consultar el GraphRAG legal de Oaxaca.

Uso:
    python telegram_bot.py

Requiere TELEGRAM_BOT_TOKEN y GROQ_API_KEY en el entorno o en .env (junto a este archivo).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv(Path(__file__).resolve().parent / ".env")

import rag_engine  # noqa: E402  (después de load_dotenv para que lea GROQ_*)

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # evita registrar la URL con el token
log = logging.getLogger("telegram_bot")

TG_LIMIT = 4096
HISTORY_KEEP = 12  # mensajes por chat guardados en memoria (rag_engine usa los últimos 6)

WELCOME = (
    "Hola. Soy un asistente sobre tránsito, vialidad y movilidad en Oaxaca. "
    "Pregúntame, por ejemplo: «¿qué pasa si me remolcan el carro?» o «artículo 2 de la Ley de Movilidad».\n\n"
    "Es orientación basada en el texto de las leyes; no sustituye asesoría legal profesional.\n"
    "Comandos: /nuevo para reiniciar la conversación, /ayuda."
)


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Parte el texto en trozos <= limit, cortando por saltos de línea o espacios."""
    parts: list[str] = []
    text = text.strip() or "(sin respuesta)"
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    parts.append(text)
    return parts


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    await update.message.reply_text("Conversación reiniciada.")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    question = (msg.text or "").strip()
    if not question:
        return
    history: list[dict] = context.chat_data.setdefault("history", [])
    await context.bot.send_chat_action(msg.chat_id, ChatAction.TYPING)
    # ask_rag es síncrona y bloqueante: se ejecuta en un hilo para no frenar el bot
    answer = await asyncio.to_thread(rag_engine.ask_rag, question, list(history))
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    del history[:-HISTORY_KEEP]
    for chunk in split_message(answer):
        await msg.reply_text(chunk)  # texto plano: el Markdown del modelo no se interpreta


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Error en el bot: %s", type(context.error).__name__)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Falta TELEGRAM_BOT_TOKEN en el entorno o en .env")
    rag_engine.warmup()  # carga el grafo al arrancar
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler(["start", "ayuda", "help"], start))
    app.add_handler(CommandHandler("nuevo", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)
    log.info("Bot iniciado (modelo: %s). Ctrl+C para detener.", rag_engine.MODEL)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
