#!/usr/bin/env python3
"""
Тест агента нормализации стиля.
"""

import asyncio
from src.infrastructure.config.database import AsyncSessionLocal
from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl
from src.application.ai_services.orchestrator import AIOrchestrator


async def test_style_normalization():
    """
    Протестировать нормализацию стиля на реальной статье.
    """
    print("\n🎨 ТЕСТ АГЕНТА СТИЛИЗАЦИИ\n")
    
    # Инициализация
    orchestrator = AIOrchestrator()
    
    # Проверка Ollama
    if not orchestrator.check_ollama():
        print("❌ Ollama недоступен!")
        return
    
    print("✅ Ollama доступен\n")
    
    # Получить статью из БД
    async with AsyncSessionLocal() as session:
        repo = ArticleRepositoryImpl(session)
        
        # Получить первую статью
        articles = await repo.find_all(limit=1)
        
        if not articles:
            print("❌ Нет статей в БД. Запустите сначала парсер.")
            return
        
        article = articles[0]
        
        print(f"📄 ОРИГИНАЛЬНАЯ СТАТЬЯ:")
        print(f"   Заголовок: {article.title}")
        print(f"   Начало: {article.content[:200]}...\n")
        
        # Применить стилизацию
        print("🤖 Применяем агент нормализации стиля...\n")
        
        # Нормализовать только вступление
        normalized_intro = orchestrator.style_normalizer.normalize_intro(
            article.content or ""
        )
        
        print("=" * 70)
        print("РЕЗУЛЬТАТ НОРМАЛИЗАЦИИ:")
        print("=" * 70)
        print(normalized_intro[:500])
        print("=" * 70)
        print("")
        
        # Сравнение
        print("📊 СРАВНЕНИЕ:")
        print(f"   Оригинал: {len(article.content)} символов")
        print(f"   Нормализовано: {len(normalized_intro)} символов")
        print("")
        
        # Проверить что убрались личные обращения
        intro_lower = normalized_intro[:300].lower()
        removed = []
        
        if 'меня зовут' not in intro_lower:
            removed.append("✅ Убрано 'меня зовут'")
        if 'я расскажу' not in intro_lower:
            removed.append("✅ Убрано 'я расскажу'")
        if 'привет' not in intro_lower:
            removed.append("✅ Убрано 'привет'")
        if 'хочу поделиться' not in intro_lower:
            removed.append("✅ Убрано 'хочу поделиться'")
        
        if removed:
            print("🎯 ЧТО УЛУЧШЕНО:")
            for item in removed:
                print(f"   {item}")
        
        print("\n✅ Тест завершён!\n")


if __name__ == '__main__':
    asyncio.run(test_style_normalization())
