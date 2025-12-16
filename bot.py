import os
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Загружаем переменные из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Максимальная длина сообщения Telegram
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

# Конфигурация
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
VLLM_BASE_URL = os.getenv('VLLM_BASE_URL')
VLLM_API_KEY = os.getenv('VLLM_API_KEY')
VLLM_MODEL = os.getenv('VLLM_MODEL')

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

# Создаем OpenAI клиент для VLLM
client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY
)

def split_message(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """Разбивает длинный текст на части для отправки в Telegram"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        
        # Ищем последний перенос строки или пробел в пределах лимита
        split_pos = text.rfind('\n', 0, max_length)
        if split_pos == -1 or split_pos < max_length // 2:
            split_pos = text.rfind(' ', 0, max_length)
        if split_pos == -1 or split_pos < max_length // 2:
            split_pos = max_length
        
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    
    return chunks


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    # При старте также сбрасываем контекст диалога
    context.user_data["history"] = []
    await update.message.reply_text(
        "Привет! Отправь мне текстовое сообщение, и я отвечу через VLLM модель.\n"
        "Контекст диалога сохраняется. Чтобы сбросить контекст, используй команду /reset."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /reset — сбрасывает контекст диалога"""
    context.user_data["history"] = []
    await update.message.reply_text("Контекст диалога сброшен.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений"""
    user_message = update.message.text
    # История диалога хранится в user_data для каждого пользователя отдельно
    history = context.user_data.get("history", [])

    # Ограничиваем длину истории, чтобы она не разрасталась бесконечно
    # Храним последние 10 сообщений (и пользователя, и ассистента)
    max_history_len = 6
    if len(history) > max_history_len:
        history = history[-max_history_len:]
        context.user_data["history"] = history
    
    try:
        logging.info(f"Отправляем запрос в VLLM: {user_message[:50]}...")

        # Формируем сообщения для модели: системное сообщение + история + новое сообщение пользователя
        messages = [
            {
                "role": "system",
                "content": "Ты специалист в области металлургии. Отвечай развернуто на вопросы.",
            },
            *history,
            {"role": "user", "content": user_message},
        ]

        response = await client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}}
        )
        
        ai_response = response.choices[0].message.content.strip()
        logging.info(f"Получен ответ от VLLM: {ai_response[:50]}...")
        
        # Обновляем историю: добавляем текущее сообщение пользователя и ответ ассистента
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": ai_response})
        # Снова применяем ограничение по длине
        if len(history) > max_history_len:
            history = history[-max_history_len:]
        context.user_data["history"] = history

        # Отправляем ответ, разбивая на части если нужно
        for chunk in split_message(ai_response):
            await update.message.reply_text(chunk)
        
    except Exception as e:
        logging.error(f"Ошибка при запросе к VLLM: {e}")
        await update.message.reply_text("Извините, произошла ошибка при обработке вашего запроса. Проверьте настройки VLLM сервера.")

def main() -> None:
    """Запуск бота"""
    # Создаем приложение
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    print("Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()