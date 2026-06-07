#!/usr/bin/env python3
"""
Публикация обработанных статей в Telegram.
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

from src.infrastructure.config.database import AsyncSessionLocal
from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl
from src.infrastructure.telegram.telegram_publisher import TelegramPublisher


async def publish_articles(
    limit: int = 5,
    min_relevance: float = 7.0,
    include_images: bool = True,
    delay: int = 60
):
    """
    Опубликовать статьи в Telegram.
    
    Args:
        limit: Количество статей
        min_relevance: Минимальная релевантность
        include_images: Публиковать с изображениями
        delay: Задержка между постами (секунды)
    """
    # Загрузить .env
    load_dotenv()
    
    # Проверить настройки
    api_id = os.getenv('TELEGRAM_API_ID')
    api_hash = os.getenv('TELEGRAM_API_HASH')
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    channel = os.getenv('TELEGRAM_CHANNEL')
    
    if not all([api_id, api_hash, channel]):
        print("❌ Ошибка: Не заданы TELEGRAM_API_ID, TELEGRAM_API_HASH или TELEGRAM_CHANNEL в .env")
        print("\nДобавьте в .env:")
        print("TELEGRAM_API_ID=your_api_id")
        print("TELEGRAM_API_HASH=your_api_hash")
        print("TELEGRAM_BOT_TOKEN=your_bot_token  # или")
        print("TELEGRAM_PHONE=+1234567890          # для публикации от имени аккаунта")
        print("TELEGRAM_CHANNEL=@your_channel      # или -1001234567890")
        return
    
    print("\n📱 ПУБЛИКАЦИЯ В TELEGRAM")
    print(f"   Канал: {channel}")
    print(f"   Лимит: {limit}")
    print(f"   Мин. релевантность: {min_relevance}")
    print(f"   Изображения: {'Да' if include_images else 'Нет'}")
    print(f"   Задержка: {delay}сек\n")
    
    # Получить статьи из БД
    async with AsyncSessionLocal() as session:
        repo = ArticleRepositoryImpl(session)
        
        # Фильтр: обработанные AI, с высокой релевантностью
        articles = await repo.find_all(
            limit=limit * 2  # Берём с запасом т.к. будет фильтр
        )
        
        # Отфильтровать
        filtered = [
            a for a in articles
            if a.relevance_score and a.relevance_score >= min_relevance
            and a.editorial_title  # Обработанные AI
        ][:limit]
        
        if not filtered:
            print("❌ Нет статей для публикации")
            print(f"   Проверьте что есть статьи с relevance_score >= {min_relevance}")
            print("   Запустите: docker-compose exec api python run_full_pipeline.py 10")
            return
        
        print(f"📊 Найдено статей: {len(filtered)}\n")
    
    # Подключиться к Telegram
    publisher = TelegramPublisher(
        api_id=api_id,
        api_hash=api_hash,
        bot_token=bot_token,
        phone=os.getenv('TELEGRAM_PHONE')
    )
    
    try:
        print("🔌 Подключение к Telegram...")
        await publisher.connect()
        print("✅ Подключено!\n")
        
        # Опубликовать
        stats = await publisher.publish_batch(
            articles=filtered,
            channel=channel,
            include_images=include_images,
            min_relevance=min_relevance,
            delay=delay
        )
        
        print("\n" + "=" * 60)
        print("📊 ИТОГИ:")
        print(f"   Опубликовано: {stats['published']}")
        print(f"   Пропущено: {stats['skipped']}")
        print(f"   Ошибок: {stats['errors']}")
        print("=" * 60 + "\n")
        
    finally:
        await publisher.disconnect()


if __name__ == '__main__':
    # Параметры из командной строки
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    min_relevance = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
    include_images = sys.argv[3].lower() != 'false' if len(sys.argv) > 3 else True
    delay = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    
    asyncio.run(publish_articles(limit, min_relevance, include_images, delay))
