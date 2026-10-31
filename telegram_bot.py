"""Bot de Telegram para consultar el GraphRAG legal de Oaxaca.

Uso:
    python telegram_bot.py

Requiere TELEGRAM_BOT_TOKEN y GROQ_API_KEY en el entorno o en .env (junto a este archivo).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

import rag_engine  # noqa: E402  (después de load_dotenv para que lea GROQ_*)

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # evita registrar la URL con el token
log = logging.getLogger("telegram_bot")

TG_LIMIT = 4096
HISTORY_KEEP = 12  # mensajes por chat guardados en memoria (rag_engine usa los últimos 6)

WELCOME = (
    "Hola. Soy un asistente sobre tránsito y movilidad en el municipio de Oaxaca de Juárez.\n\n"
    "Puedes escribirme tu duda, enviar una nota de voz o elegir una de las preguntas frecuentes de abajo:\n\n"
    "⚖️ Es orientación basada en el texto de las leyes; no sustituye asesoría legal profesional."
)

FAQ_BUTTONS = [
    [InlineKeyboardButton("🚗 ¿Qué pasa si me remolcan?", callback_data="faq:que pasa si me remolcan")],
    [InlineKeyboardButton("🅿️ Multa por doble fila", callback_data="faq:multa por estacionarse en doble fila")],
    [InlineKeyboardButton("🪪 Conducir sin licencia", callback_data="faq:sancion por manejar sin licencia")],
    [InlineKeyboardButton("🍻 Sanción por alcoholímetro", callback_data="faq:sancion por conducir bajo estado de ebriedad")],
]


def faq_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(FAQ_BUTTONS)


def build_sources_keyboard(sources: list[tuple[str, str, str]]) -> InlineKeyboardMarkup | None:
    buttons = []
    seen = set()
    for node_id, ley, art in sources:
        key = (ley, art)
        if key in seen:
            continue
        seen.add(key)
        label = f"📜 Ver Art. {art} ({'Mpal' if 'Municipio' in ley else 'Estatal'})"
        # Telegram limita callback_data a 64 bytes; usamos un ID corto determinístico (20 bytes total)
        short_id = rag_engine.short_node_id(node_id)
        buttons.append([InlineKeyboardButton(label[:60], callback_data=f"art:{short_id}")])
    return InlineKeyboardMarkup(buttons) if buttons else None


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
    await update.message.reply_text(WELCOME, reply_markup=faq_keyboard())


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    await update.message.reply_text("Conversación reiniciada.", reply_markup=faq_keyboard())


async def nueva_consulta(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    await update.message.reply_text(
        "He limpiado el contexto para un nuevo caso. Puedes empezar tu consulta o elegir una opción:",
        reply_markup=faq_keyboard(),
    )


async def ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "Guía rápida:\n\n"
        "1) Describe el hecho, escribe o manda un audio de voz.\n"
        "2) Indica si es un caso de tránsito, multa, remolque, estacionamiento o alcoholímetro.\n"
        "3) Si necesitas empezar de cero, usa /nueva_consulta.\n\n"
        "O selecciona una consulta rápida abajo:"
    )
    await update.message.reply_text(msg, reply_markup=faq_keyboard())


async def articulos(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "Normativas indexadas en este bot:\n\n"
        "- Reglamento de Tránsito Metropolitano del Municipio de Oaxaca de Juárez (Prioritario)\n"
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


async def process_question(
    chat_id: int, question: str, context: ContextTypes.DEFAULT_TYPE, reply_to_message=None
) -> None:
    history: list[dict] = context.chat_data.setdefault("history", [])
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
    answer, sources = await asyncio.to_thread(rag_engine.ask_rag_with_sources, question, list(history))
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    del history[:-HISTORY_KEEP]

    chunks = split_message(answer)
    sources_markup = build_sources_keyboard(sources)
    for i, chunk in enumerate(chunks):
        # Asignar los botones al último fragmento del mensaje
        markup = sources_markup if i == len(chunks) - 1 else None
        if reply_to_message and i == 0:
            await reply_to_message.reply_text(chunk, reply_markup=markup)
        else:
            await context.bot.send_message(chat_id=chat_id, text=chunk, reply_markup=markup)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    question = (msg.text or "").strip()
    if not question:
        return
    await process_question(msg.chat_id, question, context, reply_to_message=msg)


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    voice_or_audio = msg.voice or msg.audio
    if not voice_or_audio:
        return
    await context.bot.send_chat_action(msg.chat_id, ChatAction.TYPING)
    file = await context.bot.get_file(voice_or_audio.file_id)
    buffer = io.BytesIO()
    await file.download_to_memory(buffer)
    audio_bytes = buffer.getvalue()

    filename = getattr(voice_or_audio, "file_name", None) or "voice.ogg"
    try:
        transcription = await asyncio.to_thread(
            rag_engine.transcribe_audio, audio_bytes, filename
        )
    except Exception as e:
        log.error("Error al transcribir audio: %s", e)
        await msg.reply_text("No pude procesar el mensaje de voz. Por favor escribe tu consulta en texto.")
        return

    if not transcription:
        await msg.reply_text("No alcancé a distinguir lo que dijiste en el audio. Intenta de nuevo o escribe la consulta.")
        return

    await msg.reply_text(f"🎙️ *Escuché:* «{transcription}»")
    await process_question(msg.chat_id, transcription, context, reply_to_message=msg)


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if data.startswith("faq:"):
        question = data[len("faq:"):]
        await query.message.reply_text(f"👉 *Pregunta seleccionada:* {question}")
        await process_question(query.message.chat_id, question, context)
    elif data.startswith("art:"):
        node_id = data[len("art:"):]
        node = rag_engine.get_article_by_id(node_id)
        if not node or not node.texto.strip():
            await query.message.reply_text("No se encontró el texto completo para este artículo.")
            return
        header = f"📖 *{node.ley}*\n*Artículo {node.articulo}* (Capítulo: {node.capitulo or 's/n'}):\n\n"
        full_text = header + node.texto.strip()
        for chunk in split_message(full_text):
            await query.message.reply_text(chunk)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Error en el bot (%s): %s", type(context.error).__name__, context.error)


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
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)
    log.info("Bot iniciado (modelo: %s). Ctrl+C para detener.", rag_engine.MODEL)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
