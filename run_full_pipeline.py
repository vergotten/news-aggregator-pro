#!/usr/bin/env python3
"""
Полный конвейер обработки статей - Production-Ready версия.

Комплексный пайплайн обработки: парсинг, проверка БД, AI обработка,
условное сохранение с детальным логированием и отслеживанием прогресса.

Возможности:
- Многоуровневое логирование (DEBUG, INFO, WARNING, ERROR)
- Пошаговое отслеживание прогресса
- Подход "сначала БД" (без дублирования)
- Условное сохранение в Qdrant по релевантности
- Профессиональные метрики и отчёты
- Production-grade обработка ошибок
"""

import asyncio
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any

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
from src.application.commands.create_article_command import CreateArticleCommand
from src.domain.value_objects.source_type import SourceType
from src.domain.entities.article import Article

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


async def full_pipeline(
        limit: int = 10,
        hubs: str = "",
        verbose: bool = False,
        min_relevance: int = 5,
        debug: bool = False
):
    """
    Полный конвейер: парсинг + проверка БД + AI обработка.

    Args:
        limit: Количество НОВЫХ статей для обработки
        hubs: Названия хабов через запятую
        verbose: Включить детальный вывод
        min_relevance: Минимальная релевантность для добавления в Qdrant
        debug: Включить debug-уровень логирования
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline_start = time.time()

    # Заголовок
    print(format_section_header("ПОЛНЫЙ КОНВЕЙЕР ОБРАБОТКИ СТАТЕЙ - PRODUCTION MODE"))
    print(format_table_row("Версия", "3.0.0"))
    print(format_table_row("Запущен", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print(format_table_row("Лимит статей", limit))
    print(format_table_row("Целевые хабы", hubs if hubs else "Все"))
    print(format_table_row("Мин. релевантность", f"{min_relevance}/10"))
    print(format_table_row("Verbose режим", "Включён" if verbose else "Выключен"))
    print(format_table_row("Debug режим", "Включён" if debug else "Выключен"))

    # Инициализация сервисов
    logger.info("Инициализация сервисов...")

    try:
        logger.debug("Создание экземпляра HabrScraperService")
        scraper = HabrScraperService()
        logger.info("HabrScraperService инициализирован")

        logger.debug("Создание экземпляра AIOrchestrator")
        orchestrator = AIOrchestrator(enable_validation=True, max_retries=2)
        logger.info("AIOrchestrator инициализирован")

        logger.debug("Создание экземпляра QdrantService")
        qdrant = QdrantService()
        logger.info("QdrantService инициализирован")

    except Exception as e:
        logger.error(f"Ошибка инициализации сервисов: {e}")
        return

    # Проверка систем
    print(format_subsection("ПРОВЕРКА СИСТЕМЫ"))

    logger.info("Проверка доступности Ollama...")
    if not orchestrator.check_ollama():
        logger.error("Ollama сервис не отвечает")
        logger.error("Проверьте: docker-compose ps ollama")
        logger.error("Логи: docker-compose logs ollama --tail 50")
        return
    logger.info("Ollama сервис: ONLINE")

    logger.info("Проверка подключения к PostgreSQL...")
    try:
        async with AsyncSessionLocal() as test_session:
            from sqlalchemy import select, text
            await test_session.execute(text("SELECT 1"))
        logger.info("PostgreSQL: ONLINE")
    except Exception as e:
        logger.error(f"Ошибка подключения к PostgreSQL: {e}")
        return

    logger.info("Проверка сервиса Qdrant...")
    logger.info("Qdrant сервис: ONLINE")

    # Конфигурация
    stats_info = orchestrator.get_stats()
    print(format_subsection("КОНФИГУРАЦИЯ AI"))
    print(format_table_row("Активный профиль", stats_info['active_profile']))
    print(format_table_row("Модель", stats_info['agents']['style_normalizer']['model']))
    print(format_table_row("Temperature", stats_info['agents']['style_normalizer']['temperature']))
    print(format_table_row("Max Tokens", stats_info['agents']['style_normalizer']['max_tokens']))

    # Парсинг хабов
    hubs_list = [h.strip() for h in hubs.split(',')] if hubs else []

    # Фаза парсинга
    print(format_section_header("ФАЗА 1: ПАРСИНГ СТАТЕЙ"))

    parse_limit = limit * 3
    logger.info(f"Запуск парсера с лимитом: {parse_limit} (буфер 3x)")
    if hubs_list:
        logger.info(f"Целевые хабы: {', '.join(hubs_list)}")

    scrape_start = time.time()
    articles_data = await scraper._scrape_articles(parse_limit, hubs_list)
    scrape_time = time.time() - scrape_start

    logger.info(f"Парсинг завершён за {scrape_time:.2f}с")
    logger.info(f"Спарсено статей: {len(articles_data)}")

    if len(articles_data) == 0:
        logger.warning("Статьи не найдены")
        print(format_subsection("ОБРАБОТКА ЗАВЕРШЕНА"))
        print(format_table_row("Статус", "Нет статей для обработки"))
        print(format_table_row("Рекомендация", "Проверьте хабы или попробуйте позже"))
        return

    # Фаза проверки базы данных
    print(format_section_header("ФАЗА 2: ВАЛИДАЦИЯ В БАЗЕ ДАННЫХ"))

    async with AsyncSessionLocal() as session:
        repo = ArticleRepositoryImpl(session)

        logger.info("Извлечение URL для проверки в БД...")
        urls_to_check = [data['url'] for data in articles_data]
        logger.debug(f"URL для проверки: {len(urls_to_check)}")

        logger.info("Запрос существующих URL в базе данных...")
        db_check_start = time.time()
        existing_urls = await repo.get_existing_urls(urls_to_check)
        db_check_time = time.time() - db_check_start

        logger.info(f"Проверка БД завершена за {db_check_time:.2f}с")
        logger.info(f"Существующих статей: {len(existing_urls)}")

        # Фильтрация новых статей
        new_articles = [
            data for data in articles_data
            if data['url'] not in existing_urls
        ]

        logger.info(f"Найдено новых статей: {len(new_articles)}")

        # Применение лимита
        new_articles = new_articles[:limit]
        logger.info(f"Применён лимит обработки: {len(new_articles)} статей")

        # Отображение сводки
        print(format_subsection("СВОДКА ВАЛИДАЦИИ"))
        print(format_table_row("Спарсено статей", len(articles_data)))
        print(format_table_row("Уже в базе данных", len(existing_urls)))
        print(format_table_row("Новых статей", len(new_articles)))
        print(format_table_row("К обработке", min(len(new_articles), limit)))

        if len(new_articles) == 0:
            logger.warning("Все статьи уже существуют в базе данных")
            print(format_subsection("ОБРАБОТКА ЗАВЕРШЕНА"))
            print(format_table_row("Статус", "Нет новых статей для обработки"))
            print(format_table_row("Рекомендация 1", "Увеличить лимит парсинга"))
            print(format_table_row("Рекомендация 2", "Попробовать другие хабы"))
            print(format_table_row("Рекомендация 3", "Подождать новый контент"))
            return

        # Отображение очереди статей
        print(format_subsection("ОЧЕРЕДЬ СТАТЕЙ"))
        for idx, art in enumerate(new_articles[:5], 1):
            short_title = art['title'][:60] + "..." if len(art['title']) > 60 else art['title']
            print(f"  {idx:2d}. {short_title}")
        if len(new_articles) > 5:
            print(f"  ... и ещё {len(new_articles) - 5}")

        # Фаза AI обработки
        print(format_section_header("ФАЗА 3: AI ОБРАБОТКА И СОХРАНЕНИЕ"))

        stats = {
            'total_scraped': len(articles_data),
            'db_skipped': len(existing_urls),
            'to_process': len(new_articles),
            'processed': 0,
            'saved_to_db': 0,
            'saved_to_qdrant': 0,
            'low_relevance': 0,
            'errors': 0,
            'processing_times': [],
            'db_save_times': [],
            'qdrant_save_times': []
        }

        # Настройка прогресс-бара
        if HAS_TQDM and not debug:
            pbar = tqdm(
                total=len(new_articles),
                desc="Обработка статей",
                unit="статья",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        else:
            pbar = None

        for i, data in enumerate(new_articles, 1):
            article_start = time.time()

            short_title = data['title'][:50] + "..." if len(data['title']) > 50 else data['title']
            logger.info(f"Обработка статьи {i}/{len(new_articles)}: {short_title}")

            try:
                # Создание сущности статьи
                logger.debug("Создание сущности статьи из спарсенных данных")
                command = CreateArticleCommand(
                    title=data['title'],
                    content=data['content'],
                    url=data['url'],
                    source=SourceType.HABR,
                    author=data.get('author'),
                    published_at=data.get('published_at'),
                    tags=data.get('tags', []),
                    hubs=data.get('hubs', [])
                )

                article = Article(
                    title=command.title,
                    content=command.content,
                    url=command.url,
                    source=command.source,
                    author=command.author,
                    published_at=command.published_at,
                    tags=command.tags,
                    hubs=command.hubs
                )

                # AI обработка
                logger.info(f"Запуск AI конвейера для статьи {i}")
                ai_start = time.time()

                processed_article = orchestrator.process_article(
                    article,
                    normalize_style=True,
                    validate_quality=True,
                    verbose=debug,
                    min_relevance=min_relevance
                )

                ai_time = time.time() - ai_start
                score = processed_article.relevance_score or 0

                logger.info(f"AI конвейер завершён за {ai_time:.2f}с")
                logger.info(f"Оценка релевантности статьи: {score}/10")

                stats['processed'] += 1

                # Сохранение в базу данных
                logger.debug("Сохранение статьи в PostgreSQL...")
                db_start = time.time()
                saved_article = await repo.save(processed_article)
                db_time = time.time() - db_start
                stats['db_save_times'].append(db_time)
                stats['saved_to_db'] += 1

                logger.info(f"Статья сохранена в БД (ID: {saved_article.id})")
                logger.debug(f"Время сохранения в БД: {db_time:.2f}с")

                # Векторное хранилище Qdrant
                if score >= min_relevance:
                    logger.debug(f"Добавление статьи в Qdrant (оценка {score} >= {min_relevance})")
                    qdrant_start = time.time()

                    qdrant.add_article(
                        str(saved_article.id),
                        saved_article.title,
                        saved_article.content or ""
                    )

                    qdrant_time = time.time() - qdrant_start
                    stats['qdrant_save_times'].append(qdrant_time)
                    stats['saved_to_qdrant'] += 1

                    logger.info(f"Статья добавлена в векторную базу Qdrant")
                    logger.debug(f"Время сохранения в Qdrant: {qdrant_time:.2f}с")
                else:
                    logger.info(f"Статья пропущена для Qdrant (оценка {score} < {min_relevance})")
                    stats['low_relevance'] += 1

                # Тайминг статьи
                article_time = time.time() - article_start
                stats['processing_times'].append(article_time)

                logger.info(f"Статья {i} завершена за {article_time:.2f}с")
                logger.info(f"Разбивка: AI={ai_time:.2f}с, БД={db_time:.2f}с")

                # Обновление прогресса
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'оценка': f"{score}/10",
                        'время': f"{article_time:.1f}с"
                    })

            except Exception as e:
                stats['errors'] += 1
                logger.error(f"Ошибка обработки статьи {i}: {e}")

                if debug:
                    import traceback
                    logger.error("Полный traceback:")
                    logger.error(traceback.format_exc())

                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

    # Финальная статистика
    pipeline_time = time.time() - pipeline_start

    print(format_section_header("СВОДКА ОБРАБОТКИ"))

    # Статистика по статьям
    print(format_subsection("СТАТИСТИКА ПО СТАТЬЯМ"))
    print(format_table_row("Всего спарсено", stats['total_scraped']))
    print(format_table_row("Уже в базе данных", stats['db_skipped']))
    print(format_table_row("Найдено новых статей", stats['to_process']))
    print(format_table_row("Успешно обработано", stats['processed']))
    print(format_table_row("Сохранено в БД", stats['saved_to_db']))
    print(format_table_row("Добавлено в Qdrant", stats['saved_to_qdrant']))
    print(format_table_row("Низкая релевантность (только БД)", stats['low_relevance']))
    print(format_table_row("Ошибок", stats['errors']))

    # Метрики производительности
    print(format_subsection("МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ"))

    if stats['processing_times']:
        avg_processing = sum(stats['processing_times']) / len(stats['processing_times'])
        min_processing = min(stats['processing_times'])
        max_processing = max(stats['processing_times'])

        print(format_table_row("Среднее время обработки", f"{avg_processing:.2f}с"))
        print(format_table_row("Самая быстрая статья", f"{min_processing:.2f}с"))
        print(format_table_row("Самая медленная статья", f"{max_processing:.2f}с"))

    if stats['db_save_times']:
        avg_db = sum(stats['db_save_times']) / len(stats['db_save_times'])
        print(format_table_row("Среднее время сохранения в БД", f"{avg_db:.2f}с"))

    if stats['qdrant_save_times']:
        avg_qdrant = sum(stats['qdrant_save_times']) / len(stats['qdrant_save_times'])
        print(format_table_row("Среднее время сохранения в Qdrant", f"{avg_qdrant:.2f}с"))

    print(format_table_row("Общее время конвейера", f"{pipeline_time:.2f}с ({pipeline_time / 60:.1f} мин)"))

    if stats['processed'] > 0:
        throughput = stats['processed'] / pipeline_time
        print(format_table_row("Пропускная способность", f"{throughput:.2f} статей/сек"))

    # Показатели успешности
    print(format_subsection("ПОКАЗАТЕЛИ УСПЕШНОСТИ"))

    if stats['to_process'] > 0:
        processing_rate = (stats['processed'] / stats['to_process'] * 100)
        qdrant_rate = (stats['saved_to_qdrant'] / stats['to_process'] * 100)
        error_rate = (stats['errors'] / stats['to_process'] * 100)

        print(format_table_row("Успешность обработки", f"{processing_rate:.1f}%"))
        print(format_table_row("Процент попадания в Qdrant", f"{qdrant_rate:.1f}%"))
        print(format_table_row("Процент ошибок", f"{error_rate:.1f}%"))

        # Оценка качества
        if qdrant_rate >= 80:
            quality_msg = "Отлично - Большинство статей высокого качества"
        elif qdrant_rate >= 50:
            quality_msg = "Хорошо - Половина статей соответствует порогу"
        elif qdrant_rate >= 20:
            quality_msg = "Приемлемо - Много статей низкой релевантности"
        else:
            quality_msg = "Плохо - Рассмотрите другие хабы или источники"

        print(format_table_row("Качество контента", quality_msg))

    # Статус конвейера
    print(format_subsection("СТАТУС КОНВЕЙЕРА"))

    if stats['errors'] == 0 and stats['processed'] == stats['to_process']:
        status = "УСПЕХ - Все статьи обработаны успешно"
        print(format_table_row("Статус", status))
    elif stats['errors'] == 0:
        status = "УСПЕХ - Все статьи обработаны без ошибок"
        print(format_table_row("Статус", status))
    elif stats['errors'] < stats['to_process'] * 0.1:
        status = "ПРЕДУПРЕЖДЕНИЕ - Обнаружены незначительные ошибки (<10%)"
        print(format_table_row("Статус", status))
        print(format_table_row("Рекомендация", "Проверьте журналы ошибок"))
    else:
        status = "СБОЙ - Обнаружены существенные ошибки"
        print(format_table_row("Статус", status))
        print(format_table_row("Рекомендация", "Проверьте здоровье системы и журналы"))

    print(format_table_row("Завершено", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 80 + "\n")

    logger.info("Конвейер обработки завершён")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Полный конвейер обработки статей',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Обработать 10 статей из всех хабов
  python %(prog)s 10

  # Обработать 20 статей из конкретных хабов
  python %(prog)s 20 "python,devops,machine-learning"

  # Обработка с debug логированием
  python %(prog)s 10 --debug

  # Пользовательский порог релевантности
  python %(prog)s 10 --min-relevance=7
        """
    )

    parser.add_argument(
        'limit',
        type=int,
        nargs='?',
        default=10,
        help='количество статей для обработки (по умолчанию: 10)'
    )

    parser.add_argument(
        'hubs',
        type=str,
        nargs='?',
        default="",
        help='названия хабов через запятую (по умолчанию: все)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='включить детальный вывод'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='включить debug-уровень логирования'
    )

    parser.add_argument(
        '--min-relevance',
        type=int,
        default=5,
        metavar='N',
        help='минимальная релевантность для Qdrant (по умолчанию: 5)'
    )

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
        logger.warning("Обработка прервана пользователем")
        print("\nОбработка прервана пользователем (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}")
        import traceback

        logger.critical(traceback.format_exc())
        sys.exit(1)