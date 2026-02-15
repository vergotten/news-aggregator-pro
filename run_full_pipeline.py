#!/usr/bin/env python3
"""
Полный конвейер обработки статей
"""
import asyncio
import signal
import sys
import time
import logging
import os
import json
import hashlib
import psutil
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Set
import uuid
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import aiofiles

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Импорты приложения (без дублирующих определений)
from src.scrapers.habr.scraper_service import HabrScraperService
from src.application.ai_services.orchestrator import AIOrchestrator
from src.infrastructure.ai.qdrant_client import QdrantService
from src.infrastructure.config.database import AsyncSessionLocal
from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl
from src.domain.value_objects.source_type import SourceType
from src.domain.entities.article import Article
from src.config.models_config import (
    get_models_config,
    reset_models_config,
    ModelsConfig,
    OperationMode,
    RetryStrategy,
    CacheStrategy,
    CacheConfig,
    MonitoringConfig
)
from src.infrastructure.telegram.telegraph_publisher import TelegraphPublisher
from src.infrastructure.telegram.telegram_publisher import TelegramPublisher

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_datetime(value: Any) -> Optional[datetime]:
    """
    Безопасно конвертировать значение в datetime.

    Args:
        value: Строка ISO, datetime объект или None

    Returns:
        datetime объект или None
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        try:
            value = value.strip()

            # Заменяем Z на +00:00
            if value.endswith('Z'):
                value = value[:-1] + '+00:00'

            # Пробуем fromisoformat (Python 3.7+)
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass

            # Пробуем разные форматы
            for fmt in [
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%S.%f%z',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
            ]:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue

        except Exception as e:
            logger.warning(f"Не удалось распарсить дату '{value}': {e}")

    return None


class PipelineStatus(Enum):
    """Статусы выполнения конвейера."""
    INITIALIZING = "initializing"
    PARSING = "parsing"
    VALIDATING = "validating"
    PROCESSING = "processing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CACHED = "cached"


@dataclass
class SystemMetrics:
    """Метрики системы."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineMetrics:
    """Метрики производительности конвейера."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_scraped: int = 0
    processed: int = 0
    saved_to_db: int = 0
    saved_to_qdrant: int = 0
    published_to_telegraph: int = 0
    sent_to_telegram: int = 0
    low_relevance: int = 0
    errors: int = 0
    warnings: int = 0
    processing_times: List[float] = field(default_factory=list)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    status: PipelineStatus = PipelineStatus.INITIALIZING
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mode: OperationMode = OperationMode.NORMAL
    cache_hits: int = 0
    cache_misses: int = 0
    system_metrics: List[SystemMetrics] = field(default_factory=list)
    article_hashes: Set[str] = field(default_factory=set)
    failed_urls: Set[str] = field(default_factory=set)

    @property
    def duration(self) -> float:
        """Длительность выполнения в секундах."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def avg_processing_time(self) -> float:
        """Среднее время обработки одной статьи."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    @property
    def success_rate(self) -> float:
        """Процент успешной обработки."""
        if self.total_scraped == 0:
            return 0.0
        return (self.processed / self.total_scraped) * 100

    @property
    def cache_hit_rate(self) -> float:
        """Процент попаданий в кэш."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100

    def add_system_metrics(self):
        """Добавить текущие метрики системы."""
        try:
            metrics = SystemMetrics(
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent,
                memory_used_mb=psutil.virtual_memory().used / 1024 / 1024,
                disk_usage_percent=psutil.disk_usage('/').percent,
                network_io=dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {}
            )
            self.system_metrics.append(metrics)
        except Exception as e:
            logger.warning(f"Ошибка сбора системных метрик: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для логирования."""
        return {
            "correlation_id": self.correlation_id,
            "status": self.status.value,
            "mode": self.mode.value,
            "duration_seconds": self.duration,
            "total_scraped": self.total_scraped,
            "processed": self.processed,
            "saved_to_db": self.saved_to_db,
            "saved_to_qdrant": self.saved_to_qdrant,
            "published_to_telegraph": self.published_to_telegraph,
            "sent_to_telegram": self.sent_to_telegram,
            "low_relevance": self.low_relevance,
            "errors": self.errors,
            "warnings": self.warnings,
            "avg_processing_time": self.avg_processing_time,
            "success_rate": self.success_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "retry_counts": self.retry_counts,
            "failed_urls": list(self.failed_urls),
            "system_metrics": [m.to_dict() for m in self.system_metrics[-10:]]
        }


class PipelineInterrupted(Exception):
    """Исключение для прерывания конвейера."""
    pass


class PipelineConfig:
    """Конфигурация конвейера с валидацией."""

    def __init__(
            self,
            limit: int = 10,
            hubs: str = "",
            verbose: bool = False,
            min_relevance: int = 5,
            debug: bool = False,
            provider: Optional[str] = None,
            strategy: Optional[str] = None,
            no_fallback: bool = False,
            max_retries: int = 3,
            retry_delay: float = 1.0,
            max_concurrent: int = 5,
            batch_size: int = 10,
            timeout: float = 300.0,
            mode: OperationMode = OperationMode.NORMAL,
            retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
            enable_cache: bool = True,
            cache_dir: str = "cache/pipeline",
            enable_monitoring: bool = True,
            monitoring_interval: float = 30.0,
            save_failed_urls: bool = True,
            duplicate_check: bool = True,
            rate_limit: Optional[int] = None,
            health_check_interval: float = 60.0,
            publish_telegraph: bool = False,
            min_publish_score: int = 5,
            publish_telegram: bool = False
    ):
        self.limit = max(1, limit)
        self.hubs = hubs
        self.verbose = verbose
        self.min_relevance = max(1, min(10, min_relevance))
        self.debug = debug
        self.publish_telegraph = publish_telegraph
        self.publish_telegram = publish_telegram
        self.min_publish_score = max(1, min(10, min_publish_score))

        # Определяем провайдера: приоритет env > аргумент > default
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider:
            self.provider = env_provider
            if provider and provider != env_provider:
                logger.info(f"Провайдер из LLM_PROVIDER={env_provider} (аргумент --provider={provider} игнорирован)")
            else:
                logger.info(f"Провайдер из LLM_PROVIDER: {env_provider}")
        elif provider:
            self.provider = provider
            logger.info(f"Провайдер из аргумента: {provider}")
        else:
            self.provider = "openrouter"
            logger.info(f"Провайдер по умолчанию: {self.provider}")

        self.strategy = strategy
        self.no_fallback = no_fallback
        self.max_retries = max(0, max_retries)
        self.retry_delay = max(0.1, retry_delay)
        self.max_concurrent = max(1, max_concurrent)
        self.batch_size = max(1, batch_size)
        self.timeout = max(10.0, timeout)
        self.mode = mode
        self.retry_strategy = retry_strategy
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir)
        self.enable_monitoring = enable_monitoring
        self.monitoring_interval = monitoring_interval
        self.save_failed_urls = save_failed_urls
        self.duplicate_check = duplicate_check
        self.rate_limit = rate_limit
        self.health_check_interval = health_check_interval

        # Валидация провайдера
        if self.provider and self.provider not in ['groq', 'openrouter', 'google', 'ollama']:
            raise ValueError(f"Неподдерживаемый провайдер: {self.provider}")

        # Валидация стратегии
        if self.strategy and self.strategy not in ['cost_optimized', 'balanced', 'quality_focused', 'speed_focused']:
            raise ValueError(f"Неподдерживаемая стратегия: {self.strategy}")

        # Создаём директорию кэша
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_hubs_list(self) -> List[str]:
        """Получить список хабов."""
        return [h.strip() for h in self.hubs.split(',')] if self.hubs else []

    def get_cache_path(self, key: str) -> Path:
        """Получить путь к файлу кэша."""
        return self.cache_dir / f"{key}.json"

    def calculate_retry_delay(self, attempt: int) -> float:
        """Рассчитать задержку для повторной попытки."""
        if self.retry_strategy == RetryStrategy.EXPONENTIAL:
            return self.retry_delay * (2 ** attempt)
        elif self.retry_strategy == RetryStrategy.LINEAR:
            return self.retry_delay * (attempt + 1)
        elif self.retry_strategy == RetryStrategy.ADAPTIVE:
            base_delay = self.retry_delay * (attempt + 1)
            error_factor = min(2.0, 1.0 + (attempt * 0.1))
            return base_delay * error_factor
        else:  # FIXED
            return self.retry_delay


def format_section_header(title: str, char: str = "=", width: int = 80) -> str:
    return f"\n{char * width}\n{title}\n{char * width}"


def format_subsection(title: str, width: int = 80) -> str:
    return f"\n{'-' * width}\n{title}\n{'-' * width}"


def format_table_row(label: str, value: Any, width: int = 80) -> str:
    label_str = f"  {label}:"
    value_str = str(value)
    dots = width - len(label_str) - len(value_str)
    return f"{label_str}{' ' * dots}{value_str}"


def create_article_from_data(data: Dict[str, Any]) -> Article:
    """Создать объект Article из словаря данных парсера."""
    # Парсим дату публикации из строки в datetime
    published_at = parse_datetime(data.get('published_at'))

    return Article(
        id=uuid.uuid4(),
        title=data.get('title', ''),
        content=data.get('content', ''),
        url=data.get('url', ''),
        source=SourceType.HABR,
        author=data.get('author'),
        published_at=published_at,  # Теперь это datetime или None
        tags=data.get('tags', []),
        hubs=data.get('hubs', [])
    )


def get_article_hash(article_data: Dict[str, Any]) -> str:
    """Получить хэш статьи для проверки дубликатов."""
    content = f"{article_data.get('title', '')}{article_data.get('content', '')}{article_data.get('url', '')}"
    return hashlib.md5(content.encode()).hexdigest()


async def load_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
    """Загрузить данные из кэша."""
    if not cache_path.exists():
        return None
    try:
        async with aiofiles.open(cache_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        logger.warning(f"Ошибка загрузки кэша {cache_path}: {e}")
        return None


async def save_cache(cache_path: Path, data: Dict[str, Any]):
    """Сохранить данные в кэш с поддержкой сериализации сложных типов."""
    try:
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Type {type(obj).__name__} not serializable")

        async with aiofiles.open(cache_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
                default=default_serializer
            ))
    except Exception as e:
        logger.warning(f"Ошибка сохранения кэша {cache_path}: {e}")


async def check_llm_provider(config: PipelineConfig, metrics: PipelineMetrics) -> bool:
    """Проверить доступность LLM провайдера с повторными попытками."""
    provider_name = config.provider
    if not provider_name:
        logger.error(f"[{metrics.correlation_id}] Провайдер не указан")
        return False

    logger.info(f"[{metrics.correlation_id}] Проверка LLM провайдера: {provider_name.upper()}")

    for attempt in range(config.max_retries + 1):
        try:
            # Проверка API ключей
            api_key_checks = {
                "openrouter": ("OPENROUTER_API_KEY", "sk-or-"),
                "groq": ("GROQ_API_KEY", "gsk_"),
                "google": ("GOOGLE_API_KEY", "AIza"),
            }
            if provider_name in api_key_checks:
                env_var, prefix = api_key_checks[provider_name]
                api_key = os.getenv(env_var)
                if not api_key:
                    logger.error(f"[{metrics.correlation_id}] {env_var} не установлен!")
                    logger.error(f"[{metrics.correlation_id}] Установите в .env или передайте: export {env_var}=...")
                    return False
                if "YOUR-KEY-HERE" in api_key or len(api_key) < 10:
                    logger.error(f"[{metrics.correlation_id}] Невалидный {env_var}")
                    return False
                logger.info(f"[{metrics.correlation_id}] ✓ {env_var}: {api_key[:20]}...")

            # Проверка Ollama
            if provider_name == "ollama":
                models_cfg = get_models_config(provider="ollama")
                ollama_url = models_cfg.get_ollama_base_url()
                ollama_model = models_cfg.get_ollama_model()
                logger.info(f"[{metrics.correlation_id}] Ollama URL: {ollama_url}")
                logger.info(f"[{metrics.correlation_id}] Ollama модель: {ollama_model}")

                # Тестовое подключение к Ollama
                try:
                    import requests
                    response = requests.get(f"{ollama_url}/api/tags", timeout=10)
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        if models:
                            # Проверяем, доступна ли указанная модель
                            model_available = any(ollama_model in model.get("name", "") for model in models)
                            if model_available:
                                logger.info(f"[{metrics.correlation_id}] ✓ Модель {ollama_model} доступна в Ollama")
                            else:
                                logger.warning(
                                    f"[{metrics.correlation_id}] ⚠️ Модель {ollama_model} не найдена. Доступные модели: {', '.join([m.get('name', '') for m in models])}")
                                return False
                        else:
                            logger.warning(f"[{metrics.correlation_id}] ⚠️ Ollama доступен, но нет моделей")
                            return False
                    else:
                        logger.error(f"[{metrics.correlation_id}] Ollama недоступен: {response.status_code}")
                        if attempt == config.max_retries:
                            return False
                        continue
                except Exception as e:
                    logger.error(f"[{metrics.correlation_id}] Ошибка подключения к Ollama: {e}")
                    if attempt == config.max_retries:
                        return False
                    delay = config.calculate_retry_delay(attempt)
                    logger.warning(f"[{metrics.correlation_id}] Повтор через {delay:.1f}с...")
                    await asyncio.sleep(delay)
                    continue

            # Тестовое создание провайдера
            models_config = get_models_config(
                provider=provider_name,
                strategy=config.strategy,
                enable_fallback=not config.no_fallback
            )

            try:
                test_config = models_config.get_llm_config("classifier")
                from src.infrastructure.ai.llm_provider import LLMProviderFactory
                provider = LLMProviderFactory.create(test_config)

                # Тестовый запрос
                if provider_name == "ollama":
                    # Для Ollama мы просто проверяем, что можем создать провайдер
                    # так как мы уже проверили подключение выше
                    logger.info(f"[{metrics.correlation_id}] ✓ {provider_name.upper()} провайдер OK")
                    return True
                else:
                    # Для других провайдеров делаем тестовый запрос
                    test_response = provider.generate("Test", temperature=0.1, max_tokens=10)
                    logger.info(f"[{metrics.correlation_id}] ✓ {provider_name.upper()} провайдер OK")
                    return True
            except Exception as e:
                logger.error(f"[{metrics.correlation_id}] Ошибка создания провайдера: {e}")
                if attempt == config.max_retries:
                    return False

        except Exception as e:
            if attempt < config.max_retries:
                delay = config.calculate_retry_delay(attempt)
                logger.warning(
                    f"[{metrics.correlation_id}] Попытка {attempt + 1}/{config.max_retries + 1} не удалась: {e}")
                logger.warning(f"[{metrics.correlation_id}] Повтор через {delay:.1f}с...")
                await asyncio.sleep(delay)
                metrics.retry_counts[provider_name] = metrics.retry_counts.get(provider_name, 0) + 1
            else:
                logger.error(f"[{metrics.correlation_id}] Ошибка провайдера {provider_name}: {e}")
                return False
    return False


@asynccontextmanager
async def database_session():
    """Контекстный менеджер для сессии базы данных."""
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise
    finally:
        await session.close()


async def process_article_batch(
        articles_data: List[Dict[str, Any]],
        orchestrator: AIOrchestrator,
        config: PipelineConfig,
        metrics: PipelineMetrics
) -> Tuple[List[Article], int]:
    """
    Обработать пакет статей с повторными попытками.
    Returns:
        Кортеж (успешно обработанные статьи, количество ошибок)
    """
    processed_articles = []
    errors = 0

    for data in articles_data:
        # Проверка дубликатов
        if config.duplicate_check:
            article_hash = get_article_hash(data)
            if article_hash in metrics.article_hashes:
                logger.debug(f"[{metrics.correlation_id}] Пропуск дубликата: {data.get('url', '')}")
                continue
            metrics.article_hashes.add(article_hash)

        for attempt in range(config.max_retries + 1):
            try:
                start_time = time.time()
                article = create_article_from_data(data)
                processed_article = orchestrator.process_article(
                    article=article,
                    verbose=config.verbose,
                    min_relevance=config.min_relevance
                )

                if processed_article is None:
                    raise ValueError("Обработка статьи вернула None")

                # Метаданные
                if not hasattr(processed_article, 'metadata') or processed_article.metadata is None:
                    processed_article.metadata = {}
                processed_article.metadata.update({
                    'ai_summary': getattr(processed_article, 'editorial_teaser', None),
                    'editorial_title': getattr(processed_article, 'editorial_title', None),
                    'relevance_score': processed_article.relevance_score or 0,
                    'relevance_reason': getattr(processed_article, 'relevance_reason', None),
                    'is_news': getattr(processed_article, 'is_news', None),
                    'provider': config.provider,
                    'correlation_id': metrics.correlation_id,
                    'processing_attempts': attempt + 1,
                })

                processed_articles.append(processed_article)

                # Метрики
                elapsed = time.time() - start_time
                metrics.processing_times.append(elapsed)

                if config.verbose:
                    score = processed_article.relevance_score or 0
                    logger.info(
                        f"[{metrics.correlation_id}] Обработана: {processed_article.title[:50]}... "
                        f"(score: {score}/10, time: {elapsed:.1f}s)")
                break  # Успех, выходим из цикла повторов

            except Exception as e:
                if attempt < config.max_retries:
                    delay = config.calculate_retry_delay(attempt)
                    logger.warning(f"[{metrics.correlation_id}] Ошибка обработки статьи (попытка {attempt + 1}): {e}")
                    logger.warning(f"[{metrics.correlation_id}] Повтор через {delay:.1f}с...")
                    await asyncio.sleep(delay)
                    metrics.retry_counts["article_processing"] = metrics.retry_counts.get("article_processing", 0) + 1
                else:
                    logger.error(f"[{metrics.correlation_id}] Критическая ошибка обработки статьи: {e}")
                    if config.debug:
                        logger.error(f"[{metrics.correlation_id}] {traceback.format_exc()}")
                    errors += 1
                    # Сохраняем URL неудачной статьи
                    if config.save_failed_urls:
                        metrics.failed_urls.add(data.get('url', ''))
                    break

    return processed_articles, errors


async def save_articles_to_db(
        articles: List[Article],
        config: PipelineConfig,
        metrics: PipelineMetrics
) -> Tuple[List[Article], int]:
    """
    Сохранить статьи в базу данных с повторными попытками.
    Returns:
        Кортеж (сохранённые статьи, количество ошибок)
    """
    saved_articles = []
    errors = 0

    async with database_session() as session:
        repo = ArticleRepositoryImpl(session)
        for article in articles:
            for attempt in range(config.max_retries + 1):
                try:
                    db_article = await repo.save(article)
                    saved_articles.append(db_article)
                    break  # Успех, выходим из цикла повторов
                except Exception as e:
                    if attempt < config.max_retries:
                        delay = config.calculate_retry_delay(attempt)
                        logger.warning(
                            f"[{metrics.correlation_id}] Ошибка сохранения статьи в БД (попытка {attempt + 1}): {e}")
                        logger.warning(f"[{metrics.correlation_id}] Повтор через {delay:.1f}с...")
                        await asyncio.sleep(delay)
                        metrics.retry_counts["db_save"] = metrics.retry_counts.get("db_save", 0) + 1
                    else:
                        logger.error(f"[{metrics.correlation_id}] Критическая ошибка сохранения статьи в БД: {e}")
                        if config.debug:
                            logger.error(f"[{metrics.correlation_id}] {traceback.format_exc()}")
                        errors += 1
                        break
    return saved_articles, errors


async def save_articles_to_qdrant(
        articles: List[Article],
        qdrant: QdrantService,
        config: PipelineConfig,
        metrics: PipelineMetrics
) -> int:
    """
    Сохранить статьи в Qdrant с повторными попытками.
    Returns:
        Количество успешно сохранённых статей
    """
    saved_count = 0
    for article in articles:
        score = article.relevance_score or 0
        if score < config.min_relevance:
            metrics.low_relevance += 1
            continue

        for attempt in range(config.max_retries + 1):
            try:
                qdrant.add_article(str(article.id), article.title, article.content or "")
                saved_count += 1
                break  # Успех, выходим из цикла повторов
            except Exception as e:
                if attempt < config.max_retries:
                    delay = config.calculate_retry_delay(attempt)
                    logger.warning(
                        f"[{metrics.correlation_id}] Ошибка сохранения в Qdrant (попытка {attempt + 1}): {e}")
                    logger.warning(f"[{metrics.correlation_id}] Повтор через {delay:.1f}с...")
                    await asyncio.sleep(delay)
                    metrics.retry_counts["qdrant_save"] = metrics.retry_counts.get("qdrant_save", 0) + 1
                else:
                    logger.error(f"[{metrics.correlation_id}] Критическая ошибка сохранения в Qdrant: {e}")
                    if config.debug:
                        logger.error(f"[{metrics.correlation_id}] {traceback.format_exc()}")
                    break
    return saved_count


async def publish_articles_to_telegraph(
        articles: List[Article],
        config: PipelineConfig,
        metrics: PipelineMetrics
) -> int:
    """
    Опубликовать статьи на Telegraph.

    Создаёт страницы на Telegraph для статей с relevance_score >= min_publish_score.

    Returns:
        Количество успешно опубликованных на Telegraph
    """
    telegraph_pub = TelegraphPublisher()
    published_count = 0

    eligible = [
        a for a in articles
        if (a.relevance_score or 0) >= config.min_publish_score
    ]

    if not eligible:
        logger.info(
            f"[{metrics.correlation_id}] Telegraph: нет статей с score >= {config.min_publish_score}"
        )
        return 0

    logger.info(
        f"[{metrics.correlation_id}] Telegraph: публикация {len(eligible)} статей "
        f"(score >= {config.min_publish_score})"
    )

    for article in eligible:
        try:
            title = getattr(article, 'editorial_title', None) or article.title
            content = getattr(article, 'editorial_rewritten', None) or article.content or ""
            images = getattr(article, 'images', None) or []

            result = telegraph_pub.create_page(
                title=title,
                content=content,
                images=images,
                author_name=article.author,
                source_url=article.url,
            )

            if result.success and result.url:
                published_count += 1
                metrics.published_to_telegraph += 1

                # Сохраняем telegraph_url в атрибут статьи для Telegram
                article.telegraph_url = result.url

                # Обновляем telegraph_url в БД
                try:
                    async with database_session() as session:
                        from sqlalchemy import text as sa_text
                        await session.execute(
                            sa_text("""
                                UPDATE articles
                                SET telegraph_url = :url,
                                    updated_at = NOW()
                                WHERE id = :aid
                            """),
                            {"url": result.url, "aid": str(article.id)}
                        )
                    logger.info(
                        f"[{metrics.correlation_id}] Telegraph OK: {title[:50]}... → {result.url}"
                    )
                except Exception as db_err:
                    logger.warning(
                        f"[{metrics.correlation_id}] Telegraph URL сохранён, но ошибка обновления БД: {db_err}"
                    )
            else:
                logger.warning(
                    f"[{metrics.correlation_id}] Telegraph FAIL: {title[:50]}... — {result.error}"
                )

        except Exception as e:
            logger.error(
                f"[{metrics.correlation_id}] Telegraph ошибка для '{article.title[:40]}': {e}"
            )

    return published_count


async def send_articles_to_telegram(
        articles: List[Article],
        config: PipelineConfig,
        metrics: PipelineMetrics
) -> int:
    """
    Отправить статьи в Telegram-канал.

    Отправляет короткий тизер + ссылку на Telegraph (если есть).
    Только для статей с relevance_score >= min_publish_score.

    Returns:
        Количество успешно отправленных в Telegram
    """
    telegram_pub = TelegramPublisher()
    sent_count = 0

    eligible = [
        a for a in articles
        if (a.relevance_score or 0) >= config.min_publish_score
           and getattr(a, 'telegraph_url', None)
    ]

    if not eligible:
        logger.info(
            f"[{metrics.correlation_id}] Telegram: нет статей с Telegraph URL"
        )
        return 0

    logger.info(
        f"[{metrics.correlation_id}] Telegram: отправка {len(eligible)} постов"
    )

    for article in eligible:
        try:
            title = getattr(article, 'editorial_title', None) or article.title
            teaser = getattr(article, 'editorial_teaser', None) or ""
            telegraph_url = getattr(article, 'telegraph_url', None)
            tags = article.tags or article.hubs or []
            source_name = article.source.value if hasattr(article.source, 'value') else str(article.source)

            result = await telegram_pub.send_article_post(
                title=title,
                telegraph_url=telegraph_url,
                teaser=teaser,
                tags=tags,
                source_url=article.url,
                source_name=source_name,
            )

            if result.success:
                sent_count += 1
                metrics.sent_to_telegram += 1
                logger.info(
                    f"[{metrics.correlation_id}] Telegram OK: {title[:50]}... (msg_id={result.message_id})"
                )
            else:
                logger.warning(
                    f"[{metrics.correlation_id}] Telegram FAIL: {title[:50]}... — {result.error}"
                )

            # Rate limit: задержка между отправками
            await asyncio.sleep(2.0)

        except Exception as e:
            logger.error(
                f"[{metrics.correlation_id}] Telegram ошибка для '{article.title[:40]}': {e}"
            )

    return sent_count


async def monitor_system(metrics: PipelineMetrics, interval: float = 30.0):
    """Фоновая задача мониторинга системы."""
    while True:
        metrics.add_system_metrics()
        await asyncio.sleep(interval)


async def health_check(config: PipelineConfig, metrics: PipelineMetrics):
    """Периодическая проверка здоровья системы."""
    while True:
        try:
            # Проверка доступности БД
            async with AsyncSessionLocal() as session:
                from sqlalchemy import text
                await session.execute(text("SELECT 1"))
            # Проверка доступности Qdrant
            qdrant = QdrantService()
            qdrant.client.get_collections()
            logger.debug(f"[{metrics.correlation_id}] Health check: OK")
        except Exception as e:
            logger.warning(f"[{metrics.correlation_id}] Health check failed: {e}")
            metrics.warnings += 1
        await asyncio.sleep(config.health_check_interval)


async def save_failed_urls(metrics: PipelineMetrics, config: PipelineConfig):
    """Сохранить список неудачных URL."""
    if not metrics.failed_urls or not config.save_failed_urls:
        return

    failed_file = config.cache_dir / f"failed_urls_{metrics.correlation_id}.txt"
    try:
        async with aiofiles.open(failed_file, 'w', encoding='utf-8') as f:
            for url in metrics.failed_urls:
                await f.write(f"{url}\n")
        logger.info(f"[{metrics.correlation_id}] Сохранено {len(metrics.failed_urls)} неудачных URL в {failed_file}")
    except Exception as e:
        logger.warning(f"[{metrics.correlation_id}] Ошибка сохранения неудачных URL: {e}")


async def full_pipeline(
        limit: int = 10,
        hubs: str = "",
        verbose: bool = False,
        min_relevance: int = 5,
        debug: bool = False,
        provider: Optional[str] = None,
        strategy: Optional[str] = None,
        no_fallback: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrent: int = 5,
        batch_size: int = 10,
        timeout: float = 300.0,
        mode: str = "normal",
        retry_strategy: str = "exponential",
        enable_cache: bool = True,
        cache_dir: str = "cache/pipeline",
        enable_monitoring: bool = True,
        monitoring_interval: float = 30.0,
        enable_save_failed_urls: bool = True,
        duplicate_check: bool = True,
        rate_limit: Optional[int] = None,
        health_check_interval: float = 60.0,
        publish_telegraph: bool = False,
        min_publish_score: int = 7,
        publish_telegram: bool = False
):
    """
    Полный конвейер обработки статей с улучшенной обработкой ошибок.
    """
    # Инициализация конфигурации и метрик
    try:
        config = PipelineConfig(
            limit=limit,
            hubs=hubs,
            verbose=verbose,
            min_relevance=min_relevance,
            debug=debug,
            provider=provider,
            strategy=strategy,
            no_fallback=no_fallback,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_concurrent=max_concurrent,
            batch_size=batch_size,
            timeout=timeout,
            mode=OperationMode(mode),
            retry_strategy=RetryStrategy(retry_strategy),
            enable_cache=enable_cache,
            cache_dir=cache_dir,
            enable_monitoring=enable_monitoring,
            monitoring_interval=monitoring_interval,
            save_failed_urls=enable_save_failed_urls,
            duplicate_check=duplicate_check,
            rate_limit=rate_limit,
            health_check_interval=health_check_interval,
            publish_telegraph=publish_telegraph,
            min_publish_score=min_publish_score,
            publish_telegram=publish_telegram
        )
        metrics = PipelineMetrics(mode=config.mode)
    except ValueError as e:
        logger.error(f"Ошибка конфигурации: {e}")
        return

    # Установка уровня логирования
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Заголовок
    print(format_section_header("ПОЛНЫЙ КОНВЕЙЕР ОБРАБОТКИ СТАТЕЙ"))
    print(format_table_row("ID выполнения", metrics.correlation_id))
    print(format_table_row("Версия", "3.1"))
    print(format_table_row("Запущен", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print(format_table_row("Режим", config.mode.value.upper()))
    print(format_table_row("Лимит статей", config.limit))
    print(format_table_row("Целевые хабы", config.hubs if config.hubs else "Все"))
    print(format_table_row("Мин. релевантность", f"{config.min_relevance}/10"))
    print(format_table_row("Макс. попыток", config.max_retries))
    print(format_table_row("Стратегия повторов", config.retry_strategy.value))
    print(format_table_row("Параллельных задач", config.max_concurrent))
    print(format_table_row("Размер пакета", config.batch_size))
    print(format_table_row("Кэширование", "ВКЛ" if config.enable_cache else "ВЫКЛ"))
    print(format_table_row("Мониторинг", "ВКЛ" if config.enable_monitoring else "ВЫКЛ"))
    print(format_table_row("Telegraph", "ВКЛ" if config.publish_telegraph else "ВЫКЛ"))
    print(format_table_row("Telegram", "ВКЛ" if config.publish_telegram else "ВЫКЛ"))
    if config.publish_telegraph or config.publish_telegram:
        print(format_table_row("Мин. score публикации", f"{config.min_publish_score}/10"))

    # Обработчик сигналов для graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        logger.info(f"[{metrics.correlation_id}] Получен сигнал {signum}, начинаем graceful shutdown...")
        shutdown_event.set()
        metrics.status = PipelineStatus.INTERRUPTED

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Фоновые задачи
    monitor_task = None
    health_task = None

    try:
        # Запуск фоновых задач
        if config.enable_monitoring:
            monitor_task = asyncio.create_task(monitor_system(metrics, config.monitoring_interval))
            health_task = asyncio.create_task(health_check(config, metrics))

        # Конфигурация LLM
        print(format_subsection("КОНФИГУРАЦИЯ LLM"))

        # Сбросить кэш если указан provider
        if provider:
            reset_models_config()

        # Получить конфигурацию
        models_config = get_models_config(
            provider=config.provider,
            strategy=config.strategy,
            enable_fallback=not config.no_fallback,  # простая инверсия
            force_new=bool(provider)
        )

        print(format_table_row("Провайдер", models_config.provider_name.upper()))
        print(format_table_row("Стратегия", models_config.strategy))
        print(format_table_row("Fallback", "ВЫКЛЮЧЕН ⚠️" if not models_config.enable_fallback else "ВКЛЮЧЁН ✓"))
        if models_config.enable_fallback:
            chain = models_config.get_fallback_providers()
            print(format_table_row("Цепочка fallback", " → ".join(chain)))
        else:
            print(format_table_row("Режим", f"Только {models_config.provider_name.upper()}"))

        # Проверка провайдера
        metrics.status = PipelineStatus.INITIALIZING
        if not await check_llm_provider(config, metrics):
            logger.error(f"[{metrics.correlation_id}] Провайдер недоступен, выход")
            metrics.status = PipelineStatus.FAILED
            return

        # Инициализация сервисов
        print(format_subsection("ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ"))
        try:
            scraper = HabrScraperService()
            logger.info(f"[{metrics.correlation_id}] ✓ HabrScraperService")

            # Передаём параметры в оркестратор
            orchestrator = AIOrchestrator(
                provider=config.provider,
                enable_fallback=not config.no_fallback  # простая инверсия
            )
            logger.info(f"[{metrics.correlation_id}] ✓ AIOrchestrator")

            qdrant = QdrantService()
            logger.info(f"[{metrics.correlation_id}] ✓ QdrantService")
        except Exception as e:
            logger.error(f"[{metrics.correlation_id}] Ошибка инициализации: {e}",
                         exc_info=True)  # ✅ Добавлено детальное логирование
            metrics.status = PipelineStatus.FAILED
            return

        # Проверка БД
        try:
            async with AsyncSessionLocal() as test_session:
                from sqlalchemy import text
                await test_session.execute(text("SELECT 1"))
            logger.info(f"[{metrics.correlation_id}] ✓ PostgreSQL: OK")
        except Exception as e:
            logger.error(f"[{metrics.correlation_id}] PostgreSQL ошибка: {e}",
                         exc_info=True)  # ✅ Добавлено детальное логирование
            metrics.status = PipelineStatus.FAILED
            return

        logger.info(f"[{metrics.correlation_id}] ✓ Qdrant: OK")

        # Парсинг
        print(format_section_header("ФАЗА 1: ПАРСИНГ"))
        metrics.status = PipelineStatus.PARSING
        hubs_list = config.get_hubs_list()
        parse_limit = config.limit * 3  # Получаем больше статей для фильтрации

        # Проверка кэша
        cache_key = f"scrapped_{hubs}_{parse_limit}_{hash(tuple(hubs_list))}"
        cache_path = config.get_cache_path(cache_key)
        articles_data = []

        if config.enable_cache and config.mode == OperationMode.NORMAL:
            cached_data = await load_cache(cache_path)
            if cached_data:
                articles_data = cached_data.get('articles', [])
                metrics.cache_hits += 1
                logger.info(f"[{metrics.correlation_id}] Загружено из кэша: {len(articles_data)} статей")
                metrics.status = PipelineStatus.CACHED
            else:
                metrics.cache_misses += 1
        else:
            metrics.cache_misses += 1

        if not articles_data:
            scrape_start = time.time()
            try:
                articles_data = await asyncio.wait_for(
                    scraper._scrape_articles(parse_limit, hubs_list),
                    timeout=config.timeout
                )
                # Сохраняем в кэш
                if config.enable_cache:
                    await save_cache(cache_path, {'articles': articles_data, 'timestamp': time.time()})
            except asyncio.TimeoutError:
                logger.error(f"[{metrics.correlation_id}] Таймаут парсинга ({config.timeout}с)")
                metrics.status = PipelineStatus.FAILED
                return
            except Exception as e:
                logger.error(f"[{metrics.correlation_id}] Ошибка парсинга: {e}",
                             exc_info=True)
                metrics.status = PipelineStatus.FAILED
                return

            scrape_time = time.time() - scrape_start
            logger.info(f"[{metrics.correlation_id}] Спарсено: {len(articles_data)} за {scrape_time:.2f}с")
            metrics.total_scraped = len(articles_data)

        if not articles_data:
            print("Статьи не найдены")
            metrics.status = PipelineStatus.COMPLETED
            return

        # Проверка БД
        print(format_section_header("ФАЗА 2: ВАЛИДАЦИЯ БД"))
        metrics.status = PipelineStatus.VALIDATING
        try:
            async with database_session() as session:
                repo = ArticleRepositoryImpl(session)
                urls = [d['url'] for d in articles_data]
                existing = await repo.get_existing_urls(urls)
                new_articles_data = [d for d in articles_data if d['url'] not in existing][:config.limit]

            print(format_table_row("Спарсено", len(articles_data)))
            print(format_table_row("В БД", len(existing)))
            print(format_table_row("Новых", len(new_articles_data)))

            if not new_articles_data:
                print("Нет новых статей")
                metrics.status = PipelineStatus.COMPLETED
                return
        except Exception as e:
            logger.error(f"[{metrics.correlation_id}] Ошибка валидации БД: {e}",
                         exc_info=True)  # ✅ Добавлено детальное логирование
            metrics.status = PipelineStatus.FAILED
            return

        # AI обработка
        print(format_section_header("ФАЗА 3: AI ОБРАБОТКА"))
        metrics.status = PipelineStatus.PROCESSING

        # Разбиваем на пакеты для обработки
        batches = [
            new_articles_data[i:i + config.batch_size]
            for i in range(0, len(new_articles_data), config.batch_size)
        ]

        pbar = tqdm(total=len(new_articles_data), desc="Обработка") if HAS_TQDM else None

        # Создаём семафор для ограничения параллелизма
        semaphore = asyncio.Semaphore(config.max_concurrent)

        async def process_batch_with_semaphore(batch):
            async with semaphore:
                # Проверяем прерывание
                if shutdown_event.is_set():
                    raise PipelineInterrupted()

                # Обрабатываем пакет
                processed_articles, batch_errors = await process_article_batch(
                    batch, orchestrator, config, metrics
                )

                # Сохраняем в БД
                saved_articles, db_errors = await save_articles_to_db(
                    processed_articles, config, metrics
                )

                # Сохраняем в Qdrant
                qdrant_count = await save_articles_to_qdrant(
                    saved_articles, qdrant, config, metrics
                )

                # Обновляем метрики
                metrics.processed += len(processed_articles)
                metrics.saved_to_db += len(saved_articles)
                metrics.saved_to_qdrant += qdrant_count
                metrics.errors += batch_errors + db_errors

                # ФАЗА 4: Публикация в Telegraph
                if config.publish_telegraph and saved_articles:
                    telegraph_count = await publish_articles_to_telegraph(
                        saved_articles, config, metrics
                    )

                # ФАЗА 5: Отправка в Telegram (после Telegraph, чтобы были URL)
                if config.publish_telegram and saved_articles:
                    telegram_count = await send_articles_to_telegram(
                        saved_articles, config, metrics
                    )

                # Обновляем прогресс
                if pbar:
                    pbar.update(len(batch))
                    avg_score = sum(a.relevance_score or 0 for a in processed_articles) / len(
                        processed_articles) if processed_articles else 0
                    pbar.set_postfix({'score': f"{avg_score:.1f}/10", 'qdrant': qdrant_count})

                return processed_articles, saved_articles, qdrant_count

        # Обрабатываем пакеты concurrently
        try:
            results = await asyncio.gather(
                *[process_batch_with_semaphore(batch) for batch in batches],
                return_exceptions=True
            )

            # Проверяем результаты на исключения
            for result in results:
                if isinstance(result, Exception):
                    if isinstance(result, PipelineInterrupted):
                        raise result
                    logger.error(f"[{metrics.correlation_id}] Ошибка обработки пакета: {result}", exc_info=True)
                    metrics.errors += 1
        except PipelineInterrupted:
            logger.info(f"[{metrics.correlation_id}] Обработка прервана")
            if pbar:
                pbar.close()
            metrics.status = PipelineStatus.INTERRUPTED
            return
        finally:
            if pbar:
                pbar.close()

    except Exception as e:
        metrics.status = PipelineStatus.FAILED
        logger.error(f"[{metrics.correlation_id}] Критическая ошибка конвейера: {e}",
                     exc_info=True)
        if debug:
            logger.error(f"[{metrics.correlation_id}] {traceback.format_exc()}")
        return
    finally:
        # Финализация
        metrics.status = PipelineStatus.FINALIZING
        metrics.end_time = time.time()

        # Сохранение неудачных URL
        await save_failed_urls(metrics, config)

        # Отмена фоновых задач
        if monitor_task:
            monitor_task.cancel()
        if health_task:
            health_task.cancel()

        # Статистика
        print(format_section_header("РЕЗУЛЬТАТЫ"))
        print(format_table_row("ID выполнения", metrics.correlation_id))
        print(format_table_row("Провайдер", models_config.provider_name.upper()))
        print(format_table_row("Fallback", "OFF" if not models_config.enable_fallback else "ON"))
        print(format_table_row("Обработано", metrics.processed))
        print(format_table_row("В БД", metrics.saved_to_db))
        print(format_table_row("В Qdrant", metrics.saved_to_qdrant))
        print(format_table_row("В Telegraph", metrics.published_to_telegraph))
        print(format_table_row("В Telegram", metrics.sent_to_telegram))
        print(format_table_row("Низкая релевантность", metrics.low_relevance))
        print(format_table_row("Ошибок", metrics.errors))
        print(format_table_row("Предупреждений", metrics.warnings))
        if metrics.processing_times:
            print(format_table_row("Среднее время", f"{metrics.avg_processing_time:.2f}с"))
            print(format_table_row("Общее время", f"{metrics.duration:.2f}с"))
            print(format_table_row("Успешность", f"{metrics.success_rate:.1f}%"))
            print(format_table_row("Hit rate кэша", f"{metrics.cache_hit_rate:.1f}%"))

        # Метрики повторов
        if metrics.retry_counts:
            print(format_subsection("ПОВТОРНЫЕ ПОПЫТКИ"))
            for operation, count in metrics.retry_counts.items():
                print(format_table_row(operation, count))

        # Системные метрики
        if metrics.system_metrics:
            print(format_subsection("СИСТЕМНЫЕ МЕТРИКИ"))
            latest = metrics.system_metrics[-1]
            print(format_table_row("CPU", f"{latest.cpu_percent:.1f}%"))
            print(format_table_row("Память", f"{latest.memory_percent:.1f}% ({latest.memory_used_mb:.0f}MB)"))
            print(format_table_row("Диск", f"{latest.disk_usage_percent:.1f}%"))

        if metrics.errors == 0:
            print(format_table_row("Статус", "✅ УСПЕХ"))
            metrics.status = PipelineStatus.COMPLETED
        else:
            print(format_table_row("Статус", "⚠️  С ОШИБКАМИ"))
            metrics.status = PipelineStatus.FAILED

        # Логируем метрики в формате JSON для систем мониторинга
        logger.info(f"[{metrics.correlation_id}] Метрики конвейера: {json.dumps(metrics.to_dict())}")
        print("=" * 80)

        # Финальное логирование метрик
        logger.info(f"[{metrics.correlation_id}] Финальные метрики: {json.dumps(metrics.to_dict())}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Pipeline обработки статей v3.1 (Production Enhanced)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
# Стандартный запуск
python %(prog)s 10

# С кэшированием и мониторингом
python %(prog)s 10 --enable-cache --enable-monitoring

# С адаптивными повторами
python %(prog)s 10 --retry-strategy adaptive --max-retries 5

# С сохранением неудачных URL
python %(prog)s 10 --save-failed-urls --duplicate-check

# Полная обработка с лимитом запросов
python %(prog)s 10 --mode full --rate-limit 60

# С публикацией в Telegraph + Telegram (score >= 7)
python %(prog)s 10 --publish

# Только Telegraph (без Telegram)
python %(prog)s 10 --telegraph

# Только Telegram (нужен Telegraph URL в БД)
python %(prog)s 10 --telegram

# Отдельно Telegraph + Telegram, мин. score 8
python %(prog)s 10 --telegraph --telegram --min-publish-score 8
"""
    )
    parser.add_argument('limit', type=int, nargs='?', default=10,
                        help='Количество статей (default: 10)')
    parser.add_argument('hubs', type=str, nargs='?', default="",
                        help='Хабы через запятую (default: все)')

    # LLM параметры
    parser.add_argument('--provider', '-p',
                        choices=['groq', 'openrouter', 'google', 'ollama'],
                        help='LLM провайдер')
    parser.add_argument('--no-fallback', action='store_true',
                        help='Отключить fallback (только указанный провайдер)')
    parser.add_argument('--strategy', '-s',
                        choices=['cost_optimized', 'balanced', 'quality_focused', 'speed_focused'],
                        help='Стратегия выбора моделей')

    # Параметры производительности
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Макс. количество повторных попыток (default: 3)')
    parser.add_argument('--retry-delay', type=float, default=1.0,
                        help='Начальная задержка между попытками, сек (default: 1.0)')
    parser.add_argument('--retry-strategy', choices=['exponential', 'linear', 'fixed', 'adaptive'],
                        default='exponential', help='Стратегия повторов (default: exponential)')
    parser.add_argument('--max-concurrent', type=int, default=5,
                        help='Макс. количество параллельных задач (default: 5)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Размер пакета для обработки (default: 10)')
    parser.add_argument('--timeout', type=float, default=300.0,
                        help='Таймаут операции, сек (default: 300.0)')

    # Режимы работы (без DRY_RUN)
    parser.add_argument('--mode', choices=['normal', 'resume', 'incremental', 'full'],
                        default='normal', help='Режим работы (default: normal)')

    # Кэширование и мониторинг
    parser.add_argument('--enable-cache', action='store_true', default=True,
                        help='Включить кэширование (default: True)')
    parser.add_argument('--cache-dir', default='cache/pipeline',
                        help='Директория для кэша (default: cache/pipeline)')
    parser.add_argument('--enable-monitoring', action='store_true', default=True,
                        help='Включить мониторинг системы (default: True)')
    parser.add_argument('--monitoring-interval', type=float, default=30.0,
                        help='Интервал мониторинга, сек (default: 30.0)')

    # Дополнительные функции
    parser.add_argument('--save-failed-urls', action='store_true', default=True,
                        help='Сохранять неудачные URL (default: True)')
    parser.add_argument('--duplicate-check', action='store_true', default=True,
                        help='Проверять дубликаты (default: True)')
    parser.add_argument('--rate-limit', type=int,
                        help='Лимит запросов в минуту')
    parser.add_argument('--health-check-interval', type=float, default=60.0,
                        help='Интервал проверки здоровья, сек (default: 60.0)')

    # Другие параметры
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Подробный вывод')
    parser.add_argument('--debug', action='store_true',
                        help='Debug режим')
    parser.add_argument('--min-relevance', type=int, default=5,
                        help='Мин. релевантность для Qdrant (default: 5)')

    # Telegraph / Telegram публикация
    parser.add_argument('--publish', action='store_true', default=False,
                        help='Публиковать в Telegraph + Telegram (shortcut для --telegraph --telegram)')
    parser.add_argument('--telegraph', action='store_true', default=False,
                        help='Публиковать статьи на Telegraph (default: False)')
    parser.add_argument('--telegram', action='store_true', default=False,
                        help='Отправлять посты в Telegram-канал (default: False)')
    parser.add_argument('--min-publish-score', type=int, default=7,
                        help='Мин. score для публикации (default: 7)')

    args = parser.parse_args()

    try:
        asyncio.run(full_pipeline(
            limit=args.limit,
            hubs=args.hubs,
            verbose=args.verbose,
            min_relevance=args.min_relevance,
            debug=args.debug,
            provider=args.provider,
            strategy=args.strategy,
            no_fallback=args.no_fallback,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            max_concurrent=args.max_concurrent,
            batch_size=args.batch_size,
            timeout=args.timeout,
            mode=args.mode,
            retry_strategy=args.retry_strategy,
            enable_cache=args.enable_cache,
            cache_dir=args.cache_dir,
            enable_monitoring=args.enable_monitoring,
            monitoring_interval=args.monitoring_interval,
            enable_save_failed_urls=args.save_failed_urls,
            duplicate_check=args.duplicate_check,
            rate_limit=args.rate_limit,
            health_check_interval=args.health_check_interval,
            publish_telegraph=args.publish or args.telegraph,
            min_publish_score=args.min_publish_score,
            publish_telegram=args.publish or args.telegram
        ))
    except KeyboardInterrupt:
        print("\n⚠️  Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}", exc_info=True)
        traceback.print_exc()
        sys.exit(1)