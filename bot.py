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

# Конфигурация
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
VLLM_BASE_URL = os.getenv('VLLM_BASE_URL', 'http://localhost:8555/v1')
VLLM_API_KEY = os.getenv('VLLM_API_KEY', 'EMPTY')
VLLM_MODEL = os.getenv('VLLM_MODEL', 'Qwen/Qwen3-14b')

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

# Создаем OpenAI клиент для VLLM
client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    await update.message.reply_text('Привет! Отправь мне текстовое сообщение, и я отвечу через VLLM модель.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений"""
    user_message = update.message.text
    
    try:
        logging.info(f"Отправляем запрос в VLLM: {user_message[:50]}...")
        
        response = await client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": "Ты полезный ассистент. Отвечай кратко и по существу."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        
        ai_response = response.choices[0].message.content.strip()
        logging.info(f"Получен ответ от VLLM: {ai_response[:50]}...")
        
        await update.message.reply_text(ai_response)
        
    except Exception as e:
        logging.error(f"Ошибка при запросе к VLLM: {e}")
        await update.message.reply_text("Извините, произошла ошибка при обработке вашего запроса. Проверьте настройки VLLM сервера.")

def main() -> None:
    """Запуск бота"""
    # Создаем приложение
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    print("Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
