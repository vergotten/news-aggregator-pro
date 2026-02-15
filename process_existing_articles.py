#!/usr/bin/env python3
"""
Process Existing Articles v4.1

AI обработка статей, уже находящихся в базе данных.

Версия 4.1.0:
- Поддержка --provider (groq, openrouter, google, ollama)
- Поддержка --no-fallback (только один провайдер)
- Поддержка --strategy (cost_optimized, balanced, quality_focused)
"""

import asyncio
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.application.ai_services.orchestrator import AIOrchestrator
from src.infrastructure.ai.qdrant_client import QdrantService
from src.infrastructure.config.database import AsyncSessionLocal
from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl
from src.config.models_config import get_models_config, reset_models_config
from sqlalchemy import select
from src.infrastructure.persistence.models import ArticleModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def format_section_header(title: str, char: str = "=", width: int = 80) -> str:
    return f"\n{char * width}\n{title}\n{char * width}"


def format_subsection(title: str, width: int = 80) -> str:
    return f"\n{'-' * width}\n{title}\n{'-' * width}"


def format_table_row(label: str, value: Any, width: int = 80) -> str:
    label_str = f"  {label}:"
    value_str = str(value)
    dots = width - len(label_str) - len(value_str)
    return f"{label_str}{' ' * dots}{value_str}"


async def process_existing_articles(
    limit: Optional[int] = None,
    days: Optional[int] = None,
    reprocess_all: bool = False,
    min_relevance: int = 5,
    debug: bool = False,
    provider: Optional[str] = None,
    strategy: Optional[str] = None,
    no_fallback: bool = False
):
    """
    Обработка существующих статей из БД.

    Args:
        limit: Макс. количество статей
        days: Только статьи за последние N дней
        reprocess_all: Переобработать все (включая уже обработанные)
        min_relevance: Мин. score для Qdrant
        debug: Debug режим
        provider: LLM провайдер (groq, openrouter, google, ollama)
        strategy: Стратегия выбора моделей
        no_fallback: Отключить fallback
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline_start = time.time()

    # Header
    print(format_section_header("AI PROCESSING PIPELINE v4.1"))
    print(format_table_row("Версия", "4.1.0"))
    print(format_table_row("Запущен", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print(format_table_row("Режим", "Переобработать всё" if reprocess_all else "Только необработанные"))
    print(format_table_row("Лимит", limit if limit else "Без лимита"))
    print(format_table_row("Период", f"Последние {days} дней" if days else "Всё время"))
    print(format_table_row("Мин. релевантность", f"{min_relevance}/10"))

    # Конфигурация LLM
    print(format_subsection("КОНФИГУРАЦИЯ LLM"))
    
    if provider:
        reset_models_config()
    
    config = get_models_config(
        provider=provider,
        strategy=strategy,
        enable_fallback=not no_fallback if no_fallback else None,
        force_new=bool(provider)
    )
    
    print(format_table_row("Провайдер", config.provider_name.upper()))
    print(format_table_row("Стратегия", config.strategy))
    print(format_table_row("Fallback", "ВЫКЛЮЧЕН ⚠️" if not config.enable_fallback else "ВКЛЮЧЁН ✓"))
    
    if config.enable_fallback:
        chain = config.get_fallback_providers()
        print(format_table_row("Цепочка fallback", " → ".join(chain)))

    # Initialize services
    print(format_subsection("ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ"))

    try:
        orchestrator = AIOrchestrator(
            provider=provider,
            strategy=strategy,
            enable_fallback=not no_fallback if no_fallback else None,
            enable_validation=True,
            max_retries=2
        )
        logger.info("✓ AIOrchestrator")

        qdrant = QdrantService()
        logger.info("✓ QdrantService")

    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")
        return

    # System checks
    print(format_subsection("ПРОВЕРКА СИСТЕМ"))

    # PostgreSQL
    try:
        async with AsyncSessionLocal() as test_session:
            await test_session.execute(select(ArticleModel).limit(1))
        logger.info("✓ PostgreSQL: OK")
    except Exception as e:
        logger.error(f"PostgreSQL ошибка: {e}")
        return

    logger.info("✓ Qdrant: OK")

    # Load articles from database
    print(format_section_header("ЗАГРУЗКА СТАТЕЙ"))

    async with AsyncSessionLocal() as session:
        repo = ArticleRepositoryImpl(session)

        # Build query
        query = select(ArticleModel)

        if not reprocess_all:
            query = query.where(ArticleModel.relevance_score.is_(None))
            logger.info("Фильтр: только необработанные")

        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            query = query.where(ArticleModel.created_at >= cutoff_date)
            logger.info(f"Фильтр по времени: >= {cutoff_date.strftime('%Y-%m-%d')}")

        query = query.order_by(ArticleModel.created_at.desc())
        if limit:
            query = query.limit(limit)

        # Execute query
        result = await session.execute(query)
        models = result.scalars().all()

        print(format_table_row("Найдено статей", len(models)))

        if len(models) == 0:
            print(format_table_row("Статус", "Нет статей для обработки"))
            print(format_table_row("Совет", "Попробуйте --reprocess-all"))
            return

        # Convert to entities
        articles = [repo._to_entity(model) for model in models]

        # Display article list
        print(format_subsection("ОЧЕРЕДЬ СТАТЕЙ"))
        for idx, art in enumerate(articles[:5], 1):
            short_title = art.title[:55] + "..." if len(art.title) > 55 else art.title
            print(f"  {idx:2d}. {short_title}")
        if len(articles) > 5:
            print(f"  ... и ещё {len(articles) - 5}")

        # Statistics
        stats = {
            'total': len(articles),
            'processed': 0,
            'saved_to_qdrant': 0,
            'low_relevance': 0,
            'errors': 0,
            'article_times': [],
            'db_save_times': [],
            'qdrant_save_times': []
        }

        # Process articles
        print(format_section_header("AI ОБРАБОТКА"))

        pbar = tqdm(total=len(articles), desc="Обработка") if HAS_TQDM and not debug else None

        for i, article in enumerate(articles, 1):
            article_start = time.time()
            short_title = article.title[:40] + "..." if len(article.title) > 40 else article.title

            try:
                logger.info(f"[{i}/{len(articles)}] {short_title}")

                # AI Processing
                ai_start = time.time()
                processed_article = orchestrator.process_article(
                    article,
                    normalize_style=True,
                    validate_quality=True,
                    verbose=debug,
                    min_relevance=min_relevance
                )
                ai_time = time.time() - ai_start

                if processed_article is None:
                    stats['errors'] += 1
                    if pbar:
                        pbar.update(1)
                    continue

                score = processed_article.relevance_score or 0

                # Database save
                db_start = time.time()
                await repo.save(processed_article)
                await session.commit()
                db_time = time.time() - db_start
                
                stats['db_save_times'].append(db_time)
                stats['processed'] += 1

                # Qdrant
                if score >= min_relevance:
                    qdrant_start = time.time()
                    qdrant.add_article(
                        str(processed_article.id),
                        processed_article.title,
                        processed_article.content or ""
                    )
                    qdrant_time = time.time() - qdrant_start
                    stats['qdrant_save_times'].append(qdrant_time)
                    stats['saved_to_qdrant'] += 1
                else:
                    stats['low_relevance'] += 1

                article_time = time.time() - article_start
                stats['article_times'].append(article_time)

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({'score': f"{score}/10", 'time': f"{article_time:.1f}s"})

                logger.debug(f"Готово: AI={ai_time:.2f}s, DB={db_time:.2f}s")

            except Exception as e:
                stats['errors'] += 1
                logger.error(f"Ошибка [{i}]: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

    # Final statistics
    pipeline_time = time.time() - pipeline_start

    print(format_section_header("РЕЗУЛЬТАТЫ"))

    print(format_subsection("СТАТИСТИКА"))
    print(format_table_row("Провайдер", config.provider_name.upper()))
    print(format_table_row("Всего статей", stats['total']))
    print(format_table_row("Обработано", stats['processed']))
    print(format_table_row("В Qdrant", stats['saved_to_qdrant']))
    print(format_table_row("Низкая релевантность", stats['low_relevance']))
    print(format_table_row("Ошибок", stats['errors']))

    print(format_subsection("ПРОИЗВОДИТЕЛЬНОСТЬ"))
    if stats['article_times']:
        avg_time = sum(stats['article_times']) / len(stats['article_times'])
        print(format_table_row("Среднее время", f"{avg_time:.2f}с"))
        print(format_table_row("Мин. время", f"{min(stats['article_times']):.2f}с"))
        print(format_table_row("Макс. время", f"{max(stats['article_times']):.2f}с"))

    print(format_table_row("Общее время", f"{pipeline_time:.2f}с ({pipeline_time/60:.1f} мин)"))

    if stats['processed'] > 0:
        throughput = stats['processed'] / pipeline_time
        print(format_table_row("Throughput", f"{throughput:.2f} статей/сек"))

    # Status
    print(format_subsection("СТАТУС"))
    if stats['errors'] == 0:
        print(format_table_row("Статус", "✅ УСПЕХ"))
    elif stats['errors'] < stats['total'] * 0.1:
        print(format_table_row("Статус", "⚠️  Незначительные ошибки (<10%)"))
    else:
        print(format_table_row("Статус", "❌ Значительные ошибки"))

    print(format_table_row("Завершён", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 80 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='AI обработка существующих статей v4.1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Обработать все необработанные статьи
  python %(prog)s

  # Обработать 10 статей с Groq
  python %(prog)s --limit 10 --provider groq

  # Groq без fallback
  python %(prog)s --limit 10 --provider groq --no-fallback

  # Переобработать статьи за неделю
  python %(prog)s --days 7 --reprocess-all --provider google

  # Локальный Ollama
  python %(prog)s --limit 5 --provider ollama --no-fallback
        """
    )

    parser.add_argument('--limit', type=int, metavar='N',
                        help='Макс. количество статей')
    parser.add_argument('--days', type=int, metavar='N',
                        help='Только статьи за последние N дней')
    parser.add_argument('--reprocess-all', action='store_true',
                        help='Переобработать все статьи')
    parser.add_argument('--min-relevance', type=int, default=5, metavar='N',
                        help='Мин. score для Qdrant (default: 5)')
    
    # LLM параметры
    parser.add_argument('--provider', '-p',
                        choices=['groq', 'openrouter', 'google', 'ollama'],
                        help='LLM провайдер')
    parser.add_argument('--no-fallback', action='store_true',
                        help='Отключить fallback (только указанный провайдер)')
    parser.add_argument('--strategy', '-s',
                        choices=['cost_optimized', 'balanced', 'quality_focused', 'speed_focused'],
                        help='Стратегия выбора моделей')
    
    parser.add_argument('--debug', action='store_true',
                        help='Debug режим')

    args = parser.parse_args()

    try:
        asyncio.run(process_existing_articles(
            limit=args.limit,
            days=args.days,
            reprocess_all=args.reprocess_all,
            min_relevance=args.min_relevance,
            debug=args.debug,
            provider=args.provider,
            strategy=args.strategy,
            no_fallback=args.no_fallback
        ))
    except KeyboardInterrupt:
        print("\n⚠️  Прервано")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
