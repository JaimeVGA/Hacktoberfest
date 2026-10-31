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
from telegram import BotCommand, Update
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
    "Hola. Soy un asistente sobre tránsito y movilidad en el municipio de Oaxaca de Juárez. "
    "Pregúntame, por ejemplo: «¿qué pasa si me remolcan el carro?» o «multa por estacionarme en doble fila».\n\n"
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
    context.chat_data["history"] = []
    await update.message.reply_text(WELCOME)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    await update.message.reply_text("Conversación reiniciada.")


async def nueva_consulta(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    await update.message.reply_text(
        "He limpiado el contexto para un nuevo caso. Puedes empezar tu consulta con la descripción del hecho."
    )


async def ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "Guía rápida:\n\n"
        "1) Describe el hecho y la fecha aproximada.\n"
        "2) Indica si es un caso de tránsito, movilidad, multa, remolque, estacionamiento o accidente.\n"
        "3) Menciona la norma o artículo si lo conoces.\n"
        "4) Si necesitas empezar de cero, usa /nueva_consulta.\n\n"
        "Ejemplos: ‘¿me pueden multar por doble fila?’ o ‘¿qué pasa si me remolcaron el carro?’"
    )
    await update.message.reply_text(msg)


async def articulos(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "Normativas indexadas en este bot:\n\n"
        "- Reglamento de Tránsito Metropolitano del Municipio de Oaxaca de Juárez\n"
        "- Ley de Movilidad para el Estado de Oaxaca\n"
        "- Ley de Tránsito y Vialidad del Estado de Oaxaca\n"
        "- Ley de Ingresos del Estado de Oaxaca para el Ejercicio Fiscal 2026\n"
        "- Documento escaneado con OCR para consulta legal auxiliar"
    )
    await update.message.reply_text(msg)


async def deslinde(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Aviso: este asistente responde con base en el texto de las normas y no sustituye asesoría legal profesional. "
        "Si el caso requiere análisis jurídico específico, consulta a un abogado o autoridad competente."
    )


async def post_init(application: Application) -> None:
    await application.bot.set_my_commands(
        [
            BotCommand("start", "Iniciar o reiniciar la conversación"),
            BotCommand("nueva_consulta", "Limpiar contexto para un nuevo caso"),
            BotCommand("ayuda", "Guía de uso y cómo redactar tu caso"),
            BotCommand("articulos", "Ver normativas indexadas"),
            BotCommand("deslinde", "Aviso de exención de responsabilidad legal"),
        ]
    )


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
    app = Application.builder().token(token).post_init(post_init).build()
    app.add_handler(CommandHandler(["start", "iniciar", "reiniciar"], start))
    app.add_handler(CommandHandler("nuevo", reset))
    app.add_handler(CommandHandler("nueva_consulta", nueva_consulta))
    app.add_handler(CommandHandler(["ayuda", "help"], ayuda))
    app.add_handler(CommandHandler("articulos", articulos))
    app.add_handler(CommandHandler("deslinde", deslinde))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)
    log.info("Bot iniciado (modelo: %s). Ctrl+C para detener.", rag_engine.MODEL)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
