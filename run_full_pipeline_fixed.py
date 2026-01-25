#!/usr/bin/env python3
"""
Полный конвейер обработки статей - Production-Ready версия с OpenRouter.

Версия 3.4.0:
- Исправлена совместимость с CreateArticleCommand (разные версии)
- Смягчённый QualityValidator
- Поддержка OpenRouter и Ollama
"""

import asyncio
import sys
import time
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

# Отслеживание прогресса
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.scrapers.habr.scraper_service import HabrScraperService
from src.application.ai_services.orchestrator import AIOrchestrator
from src.infrastructure.ai.qdrant_client import QdrantService
from src.infrastructure.config.database import AsyncSessionLocal
from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl
from src.domain.value_objects.source_type import SourceType
from src.domain.entities.article import Article
from src.config.models_config import get_models_config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def format_section_header(title: str, char: str = "=", width: int = 80) -> str:
    """Форматировать заголовок секции."""
    return f"\n{char * width}\n{title}\n{char * width}"


def format_subsection(title: str, width: int = 80) -> str:
    """Форматировать подраздел."""
    return f"\n{'-' * width}\n{title}\n{'-' * width}"


def format_table_row(label: str, value: Any, width: int = 80) -> str:
    """Форматировать строку таблицы."""
    label_str = f"  {label}:"
    value_str = str(value)
    dots = width - len(label_str) - len(value_str)
    return f"{label_str}{' ' * dots}{value_str}"


def create_article_from_data(data: Dict[str, Any]) -> Article:
    """Создать объект Article из словаря данных парсера."""
    article = Article(
        id=uuid.uuid4(),
        title=data.get('title', ''),
        content=data.get('content', ''),
        url=data.get('url', ''),
        source=SourceType.HABR,  # НЕ source_type!
        author=data.get('author'),
        published_at=data.get('published_at'),
        tags=data.get('tags', []),
        hubs=data.get('hubs', [])
    )
    return article


def check_llm_provider() -> bool:
    """Проверить доступность LLM провайдера."""
    try:
        config = get_models_config()
        provider = config.get_provider()

        logger.info(f"Проверка LLM провайдера: {provider.value}")

        if provider.value == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                logger.error("OPENROUTER_API_KEY не установлен")
                return False

            if "YOUR-KEY-HERE" in api_key:
                logger.error("Замените плейсхолдер API ключа на реальный")
                return False

            logger.info(f"OpenRouter API ключ: {api_key[:25]}...")

            try:
                from src.infrastructure.ai.llm_provider import LLMProviderFactory
                test_config = config.get_llm_config("classifier")
                test_provider = LLMProviderFactory.create(test_config)
                logger.info("✓ OpenRouter провайдер OK")
                return True
            except Exception as e:
                logger.error(f"Ошибка OpenRouter: {e}")
                return False

        elif provider.value == "ollama":
            try:
                from src.infrastructure.ai.llm_provider import LLMProviderFactory
                test_config = config.get_llm_config("classifier")
                test_provider = LLMProviderFactory.create(test_config)
                logger.info("✓ Ollama провайдер OK")
                return True
            except Exception as e:
                logger.error(f"Ollama не отвечает: {e}")
                return False

        else:
            logger.warning(f"Неизвестный провайдер: {provider.value}")
            return False

    except Exception as e:
        logger.error(f"Ошибка проверки провайдера: {e}")
        return False


async def full_pipeline(
        limit: int = 10,
        hubs: str = "",
        verbose: bool = False,
        min_relevance: int = 5,
        debug: bool = False
):
    """Полный конвейер обработки статей."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline_start = time.time()

    # Заголовок
    print(format_section_header("ПОЛНЫЙ КОНВЕЙЕР ОБРАБОТКИ СТАТЕЙ"))
    print(format_table_row("Версия", "3.3.0 (OpenRouter/Ollama)"))
    print(format_table_row("Запущен", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print(format_table_row("Лимит статей", limit))
    print(format_table_row("Целевые хабы", hubs if hubs else "Все"))
    print(format_table_row("Мин. релевантность", f"{min_relevance}/10"))

    # Инициализация сервисов
    logger.info("Инициализация сервисов...")

    try:
        scraper = HabrScraperService()
        logger.info("✓ HabrScraperService")

        orchestrator = AIOrchestrator()
        logger.info("✓ AIOrchestrator")

        qdrant = QdrantService()
        logger.info("✓ QdrantService")
    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")
        return

    # Проверка систем
    print(format_subsection("ПРОВЕРКА СИСТЕМЫ"))

    if not check_llm_provider():
        return

    config = get_models_config()
    provider = config.get_provider()
    logger.info(f"✓ LLM: {provider.value.upper()}")

    try:
        async with AsyncSessionLocal() as test_session:
            from sqlalchemy import text
            await test_session.execute(text("SELECT 1"))
        logger.info("✓ PostgreSQL: OK")
    except Exception as e:
        logger.error(f"PostgreSQL ошибка: {e}")
        return

    logger.info("✓ Qdrant: OK")

    # Конфигурация
    print(format_subsection("КОНФИГУРАЦИЯ AI"))

    profile = os.getenv("LLM_PROFILE", "free_openrouter")
    sample_config = config.get_llm_config("classifier")

    print(format_table_row("Провайдер", provider.value.upper()))
    print(format_table_row("Профиль", profile))
    print(format_table_row("Модель", sample_config.model))

    if provider.value == "openrouter":
        if "free" in profile.lower() or ":free" in sample_config.model:
            print(format_table_row("Стоимость", "🆓 БЕСПЛАТНО"))
        else:
            print(format_table_row("Стоимость", "💰 Платная модель"))

    # Парсинг
    print(format_section_header("ФАЗА 1: ПАРСИНГ"))

    hubs_list = [h.strip() for h in hubs.split(',')] if hubs else []
    parse_limit = limit * 3

    scrape_start = time.time()
    articles_data = await scraper._scrape_articles(parse_limit, hubs_list)
    scrape_time = time.time() - scrape_start

    logger.info(f"Спарсено: {len(articles_data)} за {scrape_time:.2f}с")

    if not articles_data:
        print("Статьи не найдены")
        return

    # Проверка БД
    print(format_section_header("ФАЗА 2: ВАЛИДАЦИЯ БД"))

    async with AsyncSessionLocal() as session:
        repo = ArticleRepositoryImpl(session)

        urls = [d['url'] for d in articles_data]
        existing = await repo.get_existing_urls(urls)
        new_articles_data = [d for d in articles_data if d['url'] not in existing][:limit]

        print(format_table_row("Спарсено", len(articles_data)))
        print(format_table_row("В БД", len(existing)))
        print(format_table_row("Новых", len(new_articles_data)))

        if not new_articles_data:
            print("Нет новых статей")
            return

        # AI обработка
        print(format_section_header("ФАЗА 3: AI ОБРАБОТКА"))

        stats = {
            'total_scraped': len(articles_data),
            'processed': 0,
            'saved_to_db': 0,
            'saved_to_qdrant': 0,
            'low_relevance': 0,
            'errors': 0,
            'processing_times': []
        }

        pbar = tqdm(total=len(new_articles_data), desc="Обработка") if HAS_TQDM else None

        for i, data in enumerate(new_articles_data, 1):
            try:
                start = time.time()

                # Создаём объект Article
                article = create_article_from_data(data)

                # AI обработка - передаём объект Article!
                processed_article = orchestrator.process_article(
                    article=article,
                    verbose=verbose,
                    min_relevance=min_relevance
                )

                if processed_article is None:
                    stats['errors'] += 1
                    if pbar:
                        pbar.update(1)
                    continue

                score = processed_article.relevance_score or 0
                stats['processed'] += 1

                # Сохранение в БД
                # Добавляем AI метаданные в processed_article
                if not hasattr(processed_article, 'metadata') or processed_article.metadata is None:
                    processed_article.metadata = {}

                processed_article.metadata.update({
                    'ai_summary': processed_article.editorial_teaser if hasattr(processed_article,
                                                                                'editorial_teaser') else None,
                    'editorial_title': processed_article.editorial_title if hasattr(processed_article,
                                                                                    'editorial_title') else None,
                    'relevance_score': score,
                    'relevance_reason': processed_article.relevance_reason if hasattr(processed_article,
                                                                                      'relevance_reason') else None,
                    'is_news': processed_article.is_news if hasattr(processed_article, 'is_news') else None,
                })

                # repo.save принимает Article напрямую
                db_article = await repo.save(processed_article)
                await session.commit()
                stats['saved_to_db'] += 1

                # Qdrant
                if score >= min_relevance:
                    qdrant.add_article(str(db_article.id), db_article.title, db_article.content or "")
                    stats['saved_to_qdrant'] += 1
                else:
                    stats['low_relevance'] += 1

                elapsed = time.time() - start
                stats['processing_times'].append(elapsed)

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({'score': f"{score}/10", 'time': f"{elapsed:.1f}s"})

                if verbose:
                    print(f"\n   [{i}] {processed_article.title[:50]}...")
                    print(f"       Score: {score}/10 | Teaser: {(processed_article.editorial_teaser or '')[:60]}...")

            except Exception as e:
                stats['errors'] += 1
                logger.error(f"Ошибка {i}: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

    # Статистика
    total_time = time.time() - pipeline_start

    print(format_section_header("РЕЗУЛЬТАТЫ"))
    print(format_table_row("Обработано", stats['processed']))
    print(format_table_row("В БД", stats['saved_to_db']))
    print(format_table_row("В Qdrant", stats['saved_to_qdrant']))
    print(format_table_row("Низкая релевантность", stats['low_relevance']))
    print(format_table_row("Ошибок", stats['errors']))

    if stats['processing_times']:
        avg = sum(stats['processing_times']) / len(stats['processing_times'])
        print(format_table_row("Среднее время", f"{avg:.2f}с"))

    print(format_table_row("Общее время", f"{total_time:.2f}с"))

    if stats['errors'] == 0:
        print(format_table_row("Статус", "✅ УСПЕХ"))
    else:
        print(format_table_row("Статус", "⚠️  С ОШИБКАМИ"))

    print("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pipeline обработки статей')
    parser.add_argument('limit', type=int, nargs='?', default=10)
    parser.add_argument('hubs', type=str, nargs='?', default="")
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--min-relevance', type=int, default=5)

    args = parser.parse_args()

    try:
        asyncio.run(full_pipeline(
            limit=args.limit,
            hubs=args.hubs,
            verbose=args.verbose,
            min_relevance=args.min_relevance,
            debug=args.debug
        ))
    except KeyboardInterrupt:
        print("\n⚠️  Прервано")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)