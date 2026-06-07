import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from ingest_image_text import ingest_one as ingest_one_text
from ingest_images_llm import (
    ingest_single_image
)
from search_images_langgraph import search_image

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DOWNLOAD_DIR = Path(os.getenv("BOT_DOWNLOAD_DIR", "./bot_uploads")).resolve()
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODE_SEARCH = "search"
MODE_ADD_LLM = "add_llm"
MODE_ADD_TEXT = "add_text"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Бот для поиска и добавления картинок.\n\n"
        "Команды:\n"
        "/search — поиск картинки по описанию\n"
        "/add_llm — добавить фото через анализ изображения\n"
        "/add_text — добавить фото по текстовому описанию пользователя\n"
        "/cancel — сброс режима\n"
    )
    await update.message.reply_text(text)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("Режим сброшен.")


async def search_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args).strip()
    if query:
        await do_search(update, context, query)
        return

    context.user_data["mode"] = MODE_SEARCH
    await update.message.reply_text("Отправь текстовое описание картинки для поиска.")


async def add_llm_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = MODE_ADD_LLM
    await update.message.reply_text(
        "Отправь фото. Скрипт обработает изображение локально через OCR + captioning и сохранит в Qdrant."
    )


async def add_text_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = MODE_ADD_TEXT
    await update.message.reply_text(
        "Отправь фото с подписью (caption). Подпись пользователя будет использована как текст для эмбеддинга."
    )


async def do_search(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str
):

    try:

        results = await asyncio.to_thread(
            search_image,
            text,
            3
        )

        if not results:

            await update.message.reply_text(
                "Ничего не найдено."
            )
            return

        sent_count = 0

        for item in results:

            image_path = item["image_path"]

            if not os.path.exists(image_path):
                continue

            with open(image_path, "rb") as f:

                await update.message.reply_photo(
                    photo=f,
                    caption=(
                        f"Совпадение #{sent_count + 1}\n"
                        f"Score: {item['score']:.3f}"
                    )
                )

            sent_count += 1

        if sent_count == 0:

            await update.message.reply_text(
                "Файлы найдены в базе, но отсутствуют на сервере."
            )

    except Exception as e:

        

        await update.message.reply_text(
            f"Ошибка поиска: {e}"
        )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    mode = context.user_data.get("mode")

    if mode == MODE_SEARCH:
        await do_search(update, context, text)
        context.user_data.pop("mode", None)
        return

    await update.message.reply_text("Используй /search, /add_llm или /add_text.")


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = context.user_data.get("mode")
    message = update.message

   
    

    if not message.photo:
        return

    photo = message.photo[-1]
    caption = (message.caption or "").strip()
    if mode == MODE_ADD_LLM:

        file = await context.bot.get_file(
            photo.file_id
        )

        target_path = (
            DOWNLOAD_DIR /
            f"{photo.file_unique_id}.jpg"
        )

        await file.download_to_drive(
            str(target_path)
        )

        try:

            success = await asyncio.to_thread(
                ingest_single_image,
                str(target_path)
            )

            if success:

                await message.reply_text(
                    "Фото успешно добавлено в Qdrant."
                )

            else:

                await message.reply_text(
                    "Не удалось обработать изображение."
                )

        except Exception as e:

            await message.reply_text(
                f"Ошибка: {e}"
            )

        finally:

            context.user_data.pop(
                "mode",
                None
            )

        return

    if mode == MODE_ADD_TEXT:
        if not caption:
            await message.reply_text("Для этого режима отправь фото с подписью (caption).")
            return

        file = await context.bot.get_file(photo.file_id)
        target_path = DOWNLOAD_DIR / f"{photo.file_unique_id}.jpg"
        await file.download_to_drive(str(target_path))

        try:
            await asyncio.to_thread(ingest_one_text, str(target_path), caption)
            await message.reply_text("Фото добавлено по описанию пользователя.")
        except Exception as e:
            await message.reply_text(f"Ошибка при добавлении: {e}")

        context.user_data.pop("mode", None)
        return

    await message.reply_text("Сначала выбери режим: /search, /add_llm или /add_text.")


def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("add_llm", add_llm_cmd))
    app.add_handler(CommandHandler("add_text", add_text_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
