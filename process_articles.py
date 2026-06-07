#!/usr/bin/env python3
"""
MODE 2: AI Processing Only
Обрабатывает уже спарсенные статьи через AI агентов.
"""

import asyncio
import sys
from src.infrastructure.config.database import AsyncSessionLocal
from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl
from src.application.ai_services.orchestrator import AIOrchestrator
from src.domain.value_objects.article_status import ArticleStatus


async def process_pending_articles(limit: int = 10, verbose: bool = True):
    """
    Обработать необработанные статьи.
    
    Args:
        limit: Сколько статей обработать
        verbose: Показывать прогресс
    """
    print(f"\n🤖 MODE 2: AI Processing")
    print(f"   Лимит: {limit} статей\n")
    
    # Инициализация
    orchestrator = AIOrchestrator()
    
    # Проверка Ollama
    if not orchestrator.check_ollama():
        print("❌ Ollama недоступен!")
        print("   Проверьте: docker-compose ps ollama")
        return
    
    print("✅ Ollama доступен\n")
    
    # Обработка статей
    async with AsyncSessionLocal() as session:
        repo = ArticleRepositoryImpl(session)
        
        # Получить необработанные статьи
        articles = await repo.find_all(
            status=ArticleStatus.PENDING,
            limit=limit
        )
        
        if not articles:
            print("📭 Нет необработанных статей")
            return
        
        print(f"📊 Найдено {len(articles)} статей\n")
        
        processed = 0
        for i, article in enumerate(articles, 1):
            try:
                print(f"[{i}/{len(articles)}] ", end='')
                
                # AI обработка
                article = orchestrator.process_article(article, verbose=verbose)
                
                # Сохранить
                await repo.save(article)
                processed += 1
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
        
        print(f"\n✅ Обработано: {processed}/{len(articles)}")


if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(process_pending_articles(limit, verbose=True))
