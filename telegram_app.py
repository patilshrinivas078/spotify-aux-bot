import asyncio
import logging
import os
from typing import Set

from dotenv import load_dotenv
load_dotenv()

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from config import settings, validate_config
from graph import invoke_for_group, invoke_tick

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("aux_cord_bot.telegram")

# Track active Telegram group chats
_known_group_chats: Set[str] = set()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ingress handler for incoming Telegram group messages."""
    if not update.effective_chat or not update.message or not update.message.text:
        return

    group_chat_id = str(update.effective_chat.id)
    sender = (
        update.effective_user.first_name
        or update.effective_user.username
        or str(update.effective_user.id)
    )
    text = update.message.text

    _known_group_chats.add(group_chat_id)
    logger.info("Telegram msg received: group=%s sender=%s text=%r", group_chat_id, sender, text)

    # Invoke your existing LangGraph auction workflow
    try:
        result = invoke_for_group(group_chat_id, sender, text)
        reply = result.get("outgoing_reply", "")
        if reply:
            await context.bot.send_message(chat_id=group_chat_id, text=reply)
    except Exception:
        logger.exception("Error processing message for group=%s", group_chat_id)
        await context.bot.send_message(
            chat_id=group_chat_id, text="⚠️ Something went wrong processing that — try again."
        )


async def _timeout_ticker(application) -> None:
    """Background task checking if an auction timer expired for any group."""
    while True:
        try:
            await asyncio.sleep(settings.tick_interval_seconds)
            for group_chat_id in list(_known_group_chats):
                try:
                    result = invoke_tick(group_chat_id)
                    reply = result.get("outgoing_reply")
                    if reply:
                        await application.bot.send_message(chat_id=group_chat_id, text=reply)
                except Exception:
                    logger.exception("Error during tick execution for group=%s", group_chat_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Timeout ticker iteration failed")


async def post_init(application) -> None:
    """Starts the background ticker task on bot startup."""
    asyncio.create_task(_timeout_ticker(application))


def main() -> None:
    validate_config(settings, strict=False)

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "YOUR_BOT_TOKEN_HERE":
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env file!")

    app = (
        ApplicationBuilder()
        .token(token)
        .post_init(post_init)
        .build()
    )

    # Listen to all text messages
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    logger.info("Telegram Aux Cord Auction Bot started. Running polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
