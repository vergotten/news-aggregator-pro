#!/usr/bin/env python3
"""
Полный конвейер обработки статей v3.2

Два режима получения статей:
  1) Массовый (по умолчанию) — src.scrapers + async DB (требует asyncpg)
  2) --url — standalone парсинг + psycopg2 (без asyncpg)

AI-обработка (фаза 3+) работает одинаково в обоих режимах.

Изменения v3.2:
- --url для обработки конкретных статей по ссылке
- --url режим не требует asyncpg (standalone парсер + psycopg2)
- Поддержка нескольких URL через запятую или повтор --url
"""
import asyncio
import signal
import sys
import time
import logging
import os
import json
import hashlib
import re
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
import aiohttp
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# =========================================================================
# Эти импорты НЕ тянут asyncpg — они нужны в обоих режимах
# =========================================================================
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

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =========================================================================
# Standalone Habr парсер (для --url режима, без src.scrapers и asyncpg)
# Логика идентична HabrScraperService._parse_full_article()
# =========================================================================

HABR_BASE_URL = "https://habr.com"
HABR_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


def _extract_content_text(article_body) -> str:
    if not article_body:
        return ""
    for tag in article_body.find_all(['script', 'style']):
        tag.decompose()
    text = article_body.get_text(separator='\n', strip=True)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _normalize_image_url(url: str) -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if url.startswith('//'):
        url = 'https:' + url
    elif url.startswith('/'):
        url = HABR_BASE_URL + url
    if not url.startswith('http'):
        return None
    return url


def _get_best_image_url(img_tag) -> Optional[str]:
    if not img_tag:
        return None
    for attr in ('data-src', 'srcset', 'src'):
        val = img_tag.get(attr)
        if not val:
            continue
        if attr == 'srcset':
            parts = val.split(',')
            if parts:
                val = parts[-1].strip().split()[0]
        return _normalize_image_url(val)
    return None


def _extract_all_images(article_body) -> List[str]:
    if not article_body:
        return []
    images, seen = [], set()
    for figure in article_body.find_all('figure'):
        img = figure.find('img')
        if img:
            url = _get_best_image_url(img)
            if url and url not in seen:
                images.append(url)
                seen.add(url)
    for img in article_body.find_all('img'):
        if img.find_parent('figure'):
            continue
        url = _get_best_image_url(img)
        if url and url not in seen:
            images.append(url)
            seen.add(url)
    return images


async def parse_habr_article(url: str) -> Optional[Dict]:
    """Standalone парсинг одной статьи Habr без src.* зависимостей."""
    try:
        async with aiohttp.ClientSession(headers=HABR_HEADERS) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status}: {url}")
                    return None
                html = await response.text()

        soup = BeautifulSoup(html, 'html.parser')

        title_elem = soup.find('h1', class_='tm-title') or soup.find('h1', class_='tm-article-snippet__title')
        title = title_elem.find('span').text.strip() if title_elem else "Untitled"

        author_elem = soup.find('a', class_='tm-user-info__username')
        author = author_elem.text.strip() if author_elem else None

        time_elem = soup.find('time')
        published_at = None
        if time_elem and time_elem.get('datetime'):
            try:
                published_at = datetime.fromisoformat(time_elem['datetime'])
            except Exception:
                published_at = datetime.utcnow()
        else:
            published_at = datetime.utcnow()

        hubs = []
        for hub_elem in soup.find_all('a', class_='tm-publication-hub__link'):
            hub_span = hub_elem.find('span')
            if hub_span:
                hubs.append(hub_span.text.strip())

        article_body = soup.find('div', class_='tm-article-body') or soup.find('div', class_='article-formatted-body')
        if not article_body:
            logger.warning(f"Контент не найден: {url}")
            return None

        return {
            'title': title,
            'content': _extract_content_text(article_body),
            'url': url,
            'author': author,
            'published_at': published_at,
            'tags': hubs.copy(),
            'hubs': hubs,
            'images': _extract_all_images(article_body),
        }
    except Exception as e:
        logger.error(f"Ошибка парсинга {url}: {e}")
        return None


# =========================================================================
# psycopg2-based DB functions (для --url режима, без asyncpg)
# =========================================================================

def _get_db_connection_string() -> str:
    """Connection string из env (аналогично Settings, но без импорта src.*)."""
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        return db_url.replace('postgresql+asyncpg://', 'postgresql://')
    user = os.getenv('POSTGRES_USER', 'newsaggregator')
    password = os.getenv('POSTGRES_PASSWORD', 'changeme123')
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5433')
    db = os.getenv('POSTGRES_DB', 'news_aggregator')
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _get_existing_urls_sync(urls: List[str]) -> Set[str]:
    """Проверить какие URL уже в БД (psycopg2)."""
    import psycopg2
    try:
        conn = psycopg2.connect(_get_db_connection_string())
        cur = conn.cursor()
        cur.execute("SELECT url FROM articles WHERE url = ANY(%s)", (urls,))
        existing = {row[0] for row in cur.fetchall()}
        cur.close()
        conn.close()
        return existing
    except Exception as e:
        logger.warning(f"Не удалось проверить дубликаты в БД: {e}")
        return set()


def _save_article_sync(article_data: Dict, metadata: Optional[Dict] = None) -> Optional[str]:
    """
    Сохранить одну статью в БД через psycopg2.

    Returns:
        article_id если сохранена, None если дубликат/ошибка
    """
    import psycopg2
    try:
        conn = psycopg2.connect(_get_db_connection_string())
        cur = conn.cursor()
        article_id = str(uuid.uuid4())
        now = datetime.utcnow()

        pub_at = article_data.get('published_at')
        if isinstance(pub_at, datetime):
            pub_at = pub_at.isoformat()

        cur.execute("""
            INSERT INTO articles (
                id, title, content, url, source,
                author, published_at, tags, hubs, images,
                status, created_at, updated_at, article_metadata
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            ON CONFLICT (url) DO NOTHING
        """, (
            article_id,
            article_data['title'],
            article_data.get('content', ''),
            article_data['url'],
            'habr',
            article_data.get('author'),
            pub_at,
            article_data.get('tags', []),
            article_data.get('hubs', []),
            article_data.get('images', []),
            'pending',
            now,
            now,
            json.dumps(metadata or {}, ensure_ascii=False),
        ))

        saved = cur.rowcount > 0
        conn.commit()
        cur.close()
        conn.close()
        return article_id if saved else None

    except Exception as e:
        logger.error(f"DB ошибка: {e}")
        return None


def _update_article_sync(article_id: str, updates: Dict[str, Any]):
    """Обновить поля статьи в БД через psycopg2."""
    import psycopg2
    if not updates:
        return
    try:
        conn = psycopg2.connect(_get_db_connection_string())
        cur = conn.cursor()

        set_parts = []
        values = []
        for key, value in updates.items():
            set_parts.append(f"{key} = %s")
            if isinstance(value, (dict, list)):
                values.append(json.dumps(value, ensure_ascii=False))
            else:
                values.append(value)

        set_parts.append("updated_at = %s")
        values.append(datetime.utcnow())
        values.append(article_id)

        sql = f"UPDATE articles SET {', '.join(set_parts)} WHERE id = %s"
        cur.execute(sql, values)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"DB update ошибка для {article_id}: {e}")


# =========================================================================
# Общие утилиты (без зависимости от src.*)
# =========================================================================

def parse_datetime(value: Any) -> Optional[datetime]:
    """Безопасно конвертировать значение в datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            value = value.strip()
            if value.endswith('Z'):
                value = value[:-1] + '+00:00'
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
            for fmt in [
                '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S.%f%z',
                '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S', '%Y-%m-%d',
            ]:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        except Exception as e:
            logger.warning(f"Не удалось распарсить дату '{value}': {e}")
    return None


class PipelineStatus(Enum):
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
        return (self.end_time or time.time()) - self.start_time
    @property
    def avg_processing_time(self) -> float:
        return sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0
    @property
    def success_rate(self) -> float:
        return (self.processed / self.total_scraped * 100) if self.total_scraped else 0.0
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total else 0.0

    def add_system_metrics(self):
        try:
            self.system_metrics.append(SystemMetrics(
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent,
                memory_used_mb=psutil.virtual_memory().used / 1024 / 1024,
                disk_usage_percent=psutil.disk_usage('/').percent,
                network_io=dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {}
            ))
        except Exception as e:
            logger.warning(f"Ошибка сбора метрик: {e}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id, "status": self.status.value,
            "mode": self.mode.value, "duration_seconds": self.duration,
            "total_scraped": self.total_scraped, "processed": self.processed,
            "saved_to_db": self.saved_to_db, "saved_to_qdrant": self.saved_to_qdrant,
            "published_to_telegraph": self.published_to_telegraph,
            "sent_to_telegram": self.sent_to_telegram,
            "low_relevance": self.low_relevance, "errors": self.errors,
            "warnings": self.warnings, "avg_processing_time": self.avg_processing_time,
            "success_rate": self.success_rate, "cache_hit_rate": self.cache_hit_rate,
            "retry_counts": self.retry_counts, "failed_urls": list(self.failed_urls),
            "system_metrics": [m.to_dict() for m in self.system_metrics[-10:]]
        }


class PipelineConfig:
    """Конфигурация конвейера с валидацией."""

    def __init__(self, *, limit=10, hubs="", verbose=False, min_relevance=5,
                 debug=False, provider=None, strategy=None, no_fallback=False,
                 max_retries=3, retry_delay=1.0, max_concurrent=5, batch_size=10,
                 timeout=300.0, mode=OperationMode.NORMAL,
                 retry_strategy=RetryStrategy.EXPONENTIAL,
                 enable_cache=True, cache_dir="cache/pipeline",
                 enable_monitoring=True, monitoring_interval=30.0,
                 save_failed_urls=True, duplicate_check=True,
                 rate_limit=None, health_check_interval=60.0,
                 publish_telegraph=False, min_publish_score=5,
                 publish_telegram=False, urls=None):

        self.limit = max(1, limit)
        self.hubs = hubs
        self.verbose = verbose
        self.min_relevance = max(1, min(10, min_relevance))
        self.debug = debug
        self.publish_telegraph = publish_telegraph
        self.publish_telegram = publish_telegram
        self.min_publish_score = max(1, min(10, min_publish_score))
        self.urls = urls or []

        # Провайдер: env > аргумент > default
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider:
            self.provider = env_provider
        elif provider:
            self.provider = provider
        else:
            self.provider = "openrouter"
        logger.info(f"Провайдер: {self.provider}")

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

        if self.provider and self.provider not in ['groq', 'openrouter', 'google', 'ollama']:
            raise ValueError(f"Неподдерживаемый провайдер: {self.provider}")
        if self.strategy and self.strategy not in ['cost_optimized', 'balanced', 'quality_focused', 'speed_focused']:
            raise ValueError(f"Неподдерживаемая стратегия: {self.strategy}")
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_url_mode(self) -> bool:
        return bool(self.urls)

    def get_hubs_list(self) -> List[str]:
        return [h.strip() for h in self.hubs.split(',')] if self.hubs else []

    def get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def calculate_retry_delay(self, attempt: int) -> float:
        if self.retry_strategy == RetryStrategy.EXPONENTIAL:
            return self.retry_delay * (2 ** attempt)
        elif self.retry_strategy == RetryStrategy.LINEAR:
            return self.retry_delay * (attempt + 1)
        elif self.retry_strategy == RetryStrategy.ADAPTIVE:
            return self.retry_delay * (attempt + 1) * min(2.0, 1.0 + attempt * 0.1)
        return self.retry_delay


# =========================================================================
# Форматирование вывода
# =========================================================================

def fmt_header(title, char="=", w=80):
    return f"\n{char * w}\n{title}\n{char * w}"

def fmt_sub(title, w=80):
    return f"\n{'-' * w}\n{title}\n{'-' * w}"

def fmt_row(label, value, w=80):
    l = f"  {label}:"
    v = str(value)
    return f"{l}{' ' * (w - len(l) - len(v))}{v}"


def get_article_hash(data: Dict[str, Any]) -> str:
    content = f"{data.get('title', '')}{data.get('content', '')}{data.get('url', '')}"
    return hashlib.md5(content.encode()).hexdigest()


async def load_cache(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            return json.loads(await f.read())
    except Exception:
        return None


async def save_cache(path: Path, data: Dict):
    try:
        def ser(obj):
            if isinstance(obj, datetime): return obj.isoformat()
            if isinstance(obj, set): return list(obj)
            if isinstance(obj, Path): return str(obj)
            raise TypeError(f"Type {type(obj).__name__} not serializable")
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2, default=ser))
    except Exception as e:
        logger.warning(f"Ошибка кэша: {e}")


async def monitor_system(metrics, interval=30.0):
    while True:
        metrics.add_system_metrics()
        await asyncio.sleep(interval)


async def save_failed_urls_file(metrics, config):
    if not metrics.failed_urls or not config.save_failed_urls:
        return
    path = config.cache_dir / f"failed_urls_{metrics.correlation_id}.txt"
    try:
        async with aiofiles.open(path, 'w') as f:
            for url in metrics.failed_urls:
                await f.write(f"{url}\n")
        logger.info(f"Сохранено {len(metrics.failed_urls)} неудачных URL в {path}")
    except Exception as e:
        logger.warning(f"Ошибка сохранения URL: {e}")


def parse_url_args(raw: List[str]) -> List[str]:
    result = []
    for item in raw:
        for url in item.split(','):
            url = url.strip()
            if url:
                result.append(url)
    return result


# =========================================================================
# ГЛАВНЫЙ КОНВЕЙЕР
#
# Фазы 1-2 различаются по режиму:
#   --url: standalone парсер + psycopg2 (без asyncpg)
#   default: src.scrapers + async DB (требует asyncpg)
#
# Фаза 3+ (AI обработка) — одинакова. Использует AIOrchestrator
# который импортируется лениво, внутри try-блока, и НЕ тянет asyncpg.
# =========================================================================

async def full_pipeline(
        limit=10, hubs="", verbose=False, min_relevance=5, debug=False,
        provider=None, strategy=None, no_fallback=False,
        max_retries=3, retry_delay=1.0, max_concurrent=5, batch_size=10,
        timeout=300.0, mode="normal", retry_strategy="exponential",
        enable_cache=True, cache_dir="cache/pipeline",
        enable_monitoring=True, monitoring_interval=30.0,
        enable_save_failed_urls=True, duplicate_check=True,
        rate_limit=None, health_check_interval=60.0,
        publish_telegraph=False, min_publish_score=7, publish_telegram=False,
        urls=None
):
    """Полный конвейер обработки статей."""

    # --- Конфигурация ---
    try:
        config = PipelineConfig(
            limit=limit, hubs=hubs, verbose=verbose, min_relevance=min_relevance,
            debug=debug, provider=provider, strategy=strategy, no_fallback=no_fallback,
            max_retries=max_retries, retry_delay=retry_delay,
            max_concurrent=max_concurrent, batch_size=batch_size, timeout=timeout,
            mode=OperationMode(mode), retry_strategy=RetryStrategy(retry_strategy),
            enable_cache=enable_cache, cache_dir=cache_dir,
            enable_monitoring=enable_monitoring, monitoring_interval=monitoring_interval,
            save_failed_urls=enable_save_failed_urls, duplicate_check=duplicate_check,
            rate_limit=rate_limit, health_check_interval=health_check_interval,
            publish_telegraph=publish_telegraph, min_publish_score=min_publish_score,
            publish_telegram=publish_telegram, urls=urls,
        )
        metrics = PipelineMetrics(mode=config.mode)
    except ValueError as e:
        logger.error(f"Ошибка конфигурации: {e}")
        return

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Заголовок ---
    print(fmt_header("ПОЛНЫЙ КОНВЕЙЕР ОБРАБОТКИ СТАТЕЙ"))
    print(fmt_row("ID", metrics.correlation_id))
    print(fmt_row("Версия", "3.2"))
    print(fmt_row("Запущен", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print(fmt_row("Режим", "URL" if config.is_url_mode else config.mode.value.upper()))
    if config.is_url_mode:
        print(fmt_row("URL статей", len(config.urls)))
        for i, u in enumerate(config.urls, 1):
            print(fmt_row(f"  [{i}]", u[:70]))
    else:
        print(fmt_row("Лимит", config.limit))
        print(fmt_row("Хабы", config.hubs or "Все"))
    print(fmt_row("Мин. релевантность", f"{config.min_relevance}/10"))
    print(fmt_row("Макс. попыток", config.max_retries))
    print(fmt_row("Telegraph", "ВКЛ" if config.publish_telegraph else "ВЫКЛ"))
    print(fmt_row("Telegram", "ВКЛ" if config.publish_telegram else "ВЫКЛ"))

    # Graceful shutdown
    shutdown_event = asyncio.Event()
    def signal_handler(signum, frame):
        logger.info(f"[{metrics.correlation_id}] Получен сигнал {signum}, shutdown...")
        shutdown_event.set()
        metrics.status = PipelineStatus.INTERRUPTED
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    monitor_task = None
    models_config = None  # для finally-блока

    try:
        # --- Мониторинг ---
        if config.enable_monitoring:
            monitor_task = asyncio.create_task(monitor_system(metrics, config.monitoring_interval))

        # --- LLM конфигурация ---
        print(fmt_sub("КОНФИГУРАЦИЯ LLM"))
        if provider:
            reset_models_config()
        models_config = get_models_config(
            provider=config.provider, strategy=config.strategy,
            enable_fallback=not config.no_fallback, force_new=bool(provider)
        )
        print(fmt_row("Провайдер", models_config.provider_name.upper()))
        print(fmt_row("Стратегия", models_config.strategy))
        print(fmt_row("Fallback", "ВЫКЛЮЧЕН ⚠️" if not models_config.enable_fallback else "ВКЛЮЧЁН ✓"))

        # --- Проверка LLM (нужна в обоих режимах) ---
        metrics.status = PipelineStatus.INITIALIZING
        provider_ok = await _check_llm_provider(config, metrics)
        if not provider_ok:
            logger.error(f"[{metrics.correlation_id}] Провайдер недоступен")
            metrics.status = PipelineStatus.FAILED
            return

        # =====================================================================
        # ФАЗЫ 1-2: Получение статей (зависит от режима)
        # =====================================================================
        if config.is_url_mode:
            new_articles_data = await _phase12_url_mode(config, metrics)
        else:
            new_articles_data = await _phase12_bulk_mode(config, metrics)

        if not new_articles_data:
            return

        # =====================================================================
        # ФАЗА 3: AI ОБРАБОТКА (streaming, одинакова для обоих режимов)
        #
        # Ленивый импорт AIOrchestrator — он НЕ тянет asyncpg,
        # потому что оркестратор работает с LLM, а не с БД.
        # =====================================================================
        print(fmt_header("ФАЗА 3: AI ОБРАБОТКА (streaming)"))
        metrics.status = PipelineStatus.PROCESSING

        from src.application.ai_services.orchestrator import AIOrchestrator
        from src.domain.value_objects.source_type import SourceType
        from src.domain.entities.article import Article

        orchestrator = AIOrchestrator(
            provider=config.provider, enable_fallback=not config.no_fallback
        )
        logger.info(f"[{metrics.correlation_id}] ✓ AIOrchestrator")

        # Qdrant (опциональный, не критичен)
        qdrant = None
        try:
            from src.infrastructure.ai.qdrant_client import QdrantService
            qdrant = QdrantService()
            logger.info(f"[{metrics.correlation_id}] ✓ QdrantService")
        except Exception as e:
            logger.warning(f"[{metrics.correlation_id}] Qdrant недоступен: {e}")

        # Telegraph / Telegram publishers
        telegraph_pub = None
        telegram_pub = None
        if config.publish_telegraph:
            try:
                from src.infrastructure.telegram.telegraph_publisher import TelegraphPublisher
                telegraph_pub = TelegraphPublisher()
            except Exception as e:
                logger.warning(f"Telegraph недоступен: {e}")
        if config.publish_telegram:
            try:
                from src.infrastructure.telegram.telegram_publisher import TelegramPublisher
                telegram_pub = TelegramPublisher()
            except Exception as e:
                logger.warning(f"Telegram недоступен: {e}")

        total = len(new_articles_data)
        pbar = tqdm(total=total, desc="Обработка") if HAS_TQDM else None

        logger.info(
            f"[{metrics.correlation_id}] Streaming: {total} статей, "
            f"telegraph={'ON' if telegraph_pub else 'OFF'}, "
            f"telegram={'ON' if telegram_pub else 'OFF'}, "
            f"db_mode={'psycopg2' if config.is_url_mode else 'async'}"
        )

        for idx, data in enumerate(new_articles_data, 1):
            article_start = time.time()
            url = data.get('url', '')
            title_short = data.get('title', '')[:60]

            if shutdown_event.is_set():
                logger.info(f"[{metrics.correlation_id}] ⚠️ Прервано на {idx-1}/{total}")
                metrics.status = PipelineStatus.INTERRUPTED
                break

            logger.info(
                f"[{metrics.correlation_id}] {'─'*60}\n"
                f"  [{idx}/{total}] {title_short}...\n  URL: {url[:80]}"
            )

            # Дубликаты
            if config.duplicate_check:
                h = get_article_hash(data)
                if h in metrics.article_hashes:
                    logger.info(f"  [{idx}/{total}] SKIP: дубликат")
                    if pbar: pbar.update(1)
                    continue
                metrics.article_hashes.add(h)

            # --- ШАГ 1: AI обработка ---
            processed_article = None
            for attempt in range(config.max_retries + 1):
                try:
                    ai_start = time.time()
                    published_at = parse_datetime(data.get('published_at'))
                    article = Article(
                        id=uuid.uuid4(), title=data.get('title', ''),
                        content=data.get('content', ''), url=data.get('url', ''),
                        source=SourceType.HABR, author=data.get('author'),
                        published_at=published_at,
                        tags=data.get('tags', []), hubs=data.get('hubs', []),
                        images=data.get('images', []),
                    )
                    processed_article = orchestrator.process_article(
                        article=article, verbose=config.verbose,
                        min_relevance=config.min_relevance
                    )
                    if processed_article is None:
                        raise ValueError("process_article вернул None")

                    if not hasattr(processed_article, 'metadata') or processed_article.metadata is None:
                        processed_article.metadata = {}
                    processed_article.metadata.update({
                        'ai_summary': getattr(processed_article, 'editorial_teaser', None),
                        'editorial_title': getattr(processed_article, 'editorial_title', None),
                        'relevance_score': processed_article.relevance_score or 0,
                        'provider': config.provider,
                        'correlation_id': metrics.correlation_id,
                    })

                    ai_elapsed = time.time() - ai_start
                    metrics.processing_times.append(ai_elapsed)
                    metrics.processed += 1

                    score = processed_article.relevance_score or 0
                    logger.info(
                        f"  [{idx}/{total}] ✅ AI OK ({ai_elapsed:.1f}s) | "
                        f"Score: {score}/10 | "
                        f"{'НОВОСТЬ' if getattr(processed_article, 'is_news', False) else 'СТАТЬЯ'}"
                    )
                    break

                except Exception as e:
                    if attempt < config.max_retries:
                        delay = config.calculate_retry_delay(attempt)
                        logger.warning(f"  [{idx}/{total}] AI ошибка ({attempt+1}): {e}, повтор {delay:.1f}s")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"  [{idx}/{total}] ❌ AI FAIL: {e}")
                        metrics.errors += 1
                        metrics.failed_urls.add(url)

            if not processed_article:
                if pbar: pbar.update(1)
                continue

            # --- ШАГ 2: Сохранение в БД ---
            score = processed_article.relevance_score or 0
            saved_id = None

            if config.is_url_mode:
                # psycopg2 (синхронно)
                meta = processed_article.metadata or {}
                saved_id = _save_article_sync(data, metadata=meta)
                if saved_id:
                    # Обновляем AI-поля после сохранения
                    ai_updates = {
                        'is_news': getattr(processed_article, 'is_news', False),
                        'relevance_score': processed_article.relevance_score,
                        'relevance_reason': getattr(processed_article, 'relevance_reason', None),
                        'editorial_title': getattr(processed_article, 'editorial_title', None),
                        'editorial_teaser': getattr(processed_article, 'editorial_teaser', None),
                        'editorial_rewritten': getattr(processed_article, 'editorial_rewritten', None),
                        'telegram_post_text': getattr(processed_article, 'telegram_post_text', None),
                        'telegram_cover_image': getattr(processed_article, 'telegram_cover_image', None),
                        'telegraph_content_html': getattr(processed_article, 'telegraph_content_html', None),
                        'status': 'processed',
                        'article_metadata': json.dumps(meta, ensure_ascii=False),
                    }
                    _update_article_sync(saved_id, ai_updates)
                    metrics.saved_to_db += 1
                    logger.info(f"  [{idx}/{total}] ✅ DB OK (psycopg2, id={saved_id})")
                else:
                    logger.warning(f"  [{idx}/{total}] ⚠️ DB: дубликат или ошибка")
            else:
                # async (через src.*)
                from src.infrastructure.config.database import AsyncSessionLocal
                from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl
                for attempt in range(config.max_retries + 1):
                    try:
                        async with AsyncSessionLocal() as session:
                            repo = ArticleRepositoryImpl(session)
                            saved = await repo.save(processed_article)
                            await session.commit()
                        saved_id = str(saved.id)
                        metrics.saved_to_db += 1
                        logger.info(f"  [{idx}/{total}] ✅ DB OK (async, id={saved_id})")
                        break
                    except Exception as e:
                        if attempt < config.max_retries:
                            delay = config.calculate_retry_delay(attempt)
                            logger.warning(f"  [{idx}/{total}] DB ({attempt+1}): {e}, повтор {delay:.1f}s")
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"  [{idx}/{total}] ❌ DB FAIL: {e}")
                            metrics.errors += 1

            if not saved_id:
                if pbar: pbar.update(1)
                continue

            # --- ШАГ 3: Qdrant ---
            if qdrant and score >= config.min_relevance:
                try:
                    qdrant.add_article(saved_id, processed_article.title, processed_article.content or "")
                    metrics.saved_to_qdrant += 1
                    logger.info(f"  [{idx}/{total}] ✅ Qdrant OK")
                except Exception as e:
                    logger.warning(f"  [{idx}/{total}] ⚠️ Qdrant: {e}")
            elif score < config.min_relevance:
                metrics.low_relevance += 1

            # --- ШАГ 4: Telegraph ---
            telegraph_url = None
            if telegraph_pub and score >= config.min_publish_score:
                try:
                    t_title = getattr(processed_article, 'editorial_title', None) or processed_article.title
                    t_content = getattr(processed_article, 'editorial_rewritten', None) or processed_article.content or ""
                    t_images = getattr(processed_article, 'images', None) or []

                    result = telegraph_pub.create_page(
                        title=t_title, content=t_content, images=t_images,
                        author_name=processed_article.author,
                    )
                    if result.success and result.url:
                        telegraph_url = result.url
                        metrics.published_to_telegraph += 1
                        _update_article_sync(saved_id, {'telegraph_url': telegraph_url})
                        logger.info(f"  [{idx}/{total}] ✅ Telegraph OK: {telegraph_url}")
                    else:
                        logger.warning(f"  [{idx}/{total}] ⚠️ Telegraph: {getattr(result, 'error', 'unknown')}")
                except Exception as e:
                    logger.error(f"  [{idx}/{total}] ❌ Telegraph: {e}")

            # --- ШАГ 5: Telegram ---
            if telegram_pub and score >= config.min_publish_score:
                try:
                    tg_post_text = getattr(processed_article, 'telegram_post_text', None)
                    if tg_post_text and telegraph_url:
                        tg_post_text = tg_post_text.replace("📖 Читать полностью → {TELEGRAPH_URL}", f'📖 <a href="{telegraph_url}">Читать полностью →</a>')
                        logger.info(f"  [{idx}/{total}] 📱 Telegram: full post, {len(tg_post_text)} chars")
                        tg_result = await telegram_pub.send_message(tg_post_text)
                    elif telegraph_url:
                        t_title = getattr(processed_article, 'editorial_title', None) or processed_article.title
                        t_teaser = getattr(processed_article, 'editorial_teaser', None) or ""
                        t_tags = processed_article.tags or getattr(processed_article, 'hubs', []) or []
                        logger.info(f"  [{idx}/{total}] 📱 Telegram: fallback short post")
                        tg_result = await telegram_pub.send_article_post(title=t_title, telegraph_url=telegraph_url, teaser=t_teaser, tags=t_tags)
                    else:
                        logger.warning(f"  [{idx}/{total}] Telegram: no Telegraph URL, skip")
                        tg_result = None
                    if tg_result and tg_result.success:
                        metrics.sent_to_telegram += 1
                        logger.info(f"  [{idx}/{total}] ✅ Telegram OK (msg_id={tg_result.message_id})")
                    else:
                        logger.warning(f"  [{idx}/{total}] ⚠️ Telegram: {tg_result.error}")
                    await asyncio.sleep(2.0)
                except Exception as e:
                    logger.error(f"  [{idx}/{total}] ❌ Telegram: {e}")

            # Итог
            elapsed = time.time() - article_start
            logger.info(
                f"  [{idx}/{total}] DONE ({elapsed:.1f}s) | "
                f"DB:{metrics.saved_to_db} Qdrant:{metrics.saved_to_qdrant} "
                f"Tph:{metrics.published_to_telegraph} Tg:{metrics.sent_to_telegram} "
                f"Err:{metrics.errors}"
            )
            if pbar:
                pbar.update(1)
                pbar.set_postfix(score=f"{score:.0f}", db=metrics.saved_to_db)

        if pbar:
            pbar.close()

    except Exception as e:
        metrics.status = PipelineStatus.FAILED
        logger.error(f"[{metrics.correlation_id}] Критическая ошибка: {e}", exc_info=True)
        return
    finally:
        metrics.status = PipelineStatus.FINALIZING
        metrics.end_time = time.time()
        await save_failed_urls_file(metrics, config)
        if monitor_task:
            monitor_task.cancel()

        # Результаты
        print(fmt_header("РЕЗУЛЬТАТЫ"))
        print(fmt_row("ID", metrics.correlation_id))
        if models_config:
            print(fmt_row("Провайдер", models_config.provider_name.upper()))
        if config.is_url_mode:
            print(fmt_row("Режим", f"URL ({len(config.urls)} шт.)"))
        print(fmt_row("Обработано AI", metrics.processed))
        print(fmt_row("В БД", metrics.saved_to_db))
        print(fmt_row("В Qdrant", metrics.saved_to_qdrant))
        print(fmt_row("В Telegraph", metrics.published_to_telegraph))
        print(fmt_row("В Telegram", metrics.sent_to_telegram))
        print(fmt_row("Низкая релевантность", metrics.low_relevance))
        print(fmt_row("Ошибок", metrics.errors))
        if metrics.processing_times:
            print(fmt_row("Среднее время AI", f"{metrics.avg_processing_time:.2f}с"))
            print(fmt_row("Общее время", f"{metrics.duration:.2f}с"))
            print(fmt_row("Успешность", f"{metrics.success_rate:.1f}%"))

        status = "✅ УСПЕХ" if metrics.errors == 0 else "⚠️  С ОШИБКАМИ"
        print(fmt_row("Статус", status))
        metrics.status = PipelineStatus.COMPLETED if metrics.errors == 0 else PipelineStatus.FAILED
        logger.info(f"[{metrics.correlation_id}] Метрики: {json.dumps(metrics.to_dict())}")
        print("=" * 80)


# =========================================================================
# Фаза 1-2: URL режим (standalone, psycopg2)
# =========================================================================

async def _phase12_url_mode(config: PipelineConfig, metrics: PipelineMetrics) -> List[Dict]:
    """Парсинг по URL + проверка дубликатов через psycopg2."""

    print(fmt_header(f"ФАЗА 1: ПАРСИНГ ПО URL ({len(config.urls)} шт.)"))
    metrics.status = PipelineStatus.PARSING

    articles_data = []
    for i, url in enumerate(config.urls, 1):
        url = url.strip()
        if not url:
            continue
        logger.info(f"[{metrics.correlation_id}] Парсинг [{i}/{len(config.urls)}]: {url}")
        article = await parse_habr_article(url)
        if article:
            articles_data.append(article)
        else:
            metrics.failed_urls.add(url)
            metrics.errors += 1

    metrics.total_scraped = len(articles_data)
    print(fmt_row("Запрошено", len(config.urls)))
    print(fmt_row("Спарсено", len(articles_data)))

    if not articles_data:
        print("Ни одна статья не спарсена")
        metrics.status = PipelineStatus.COMPLETED
        return []

    # Проверка дубликатов
    print(fmt_header("ФАЗА 2: ПРОВЕРКА ДУБЛИКАТОВ"))
    metrics.status = PipelineStatus.VALIDATING
    existing = _get_existing_urls_sync([d['url'] for d in articles_data])
    new_data = [d for d in articles_data if d['url'] not in existing]

    print(fmt_row("Уже в БД", len(existing)))
    print(fmt_row("Новых", len(new_data)))

    if not new_data:
        print("Все статьи уже в БД")
        metrics.status = PipelineStatus.COMPLETED
    return new_data


# =========================================================================
# Фаза 1-2: Bulk режим (оригинал, через src.* / asyncpg)
# =========================================================================

async def _phase12_bulk_mode(config: PipelineConfig, metrics: PipelineMetrics) -> List[Dict]:
    """Массовый парсинг + валидация через async DB."""

    # Эти импорты тянут asyncpg — но только если мы в bulk-режиме
    from src.scrapers.habr.scraper_service import HabrScraperService
    from src.infrastructure.config.database import AsyncSessionLocal
    from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl

    print(fmt_header("ФАЗА 1: ПАРСИНГ"))
    metrics.status = PipelineStatus.PARSING

    scraper = HabrScraperService()
    hubs_list = config.get_hubs_list()
    parse_limit = config.limit * 3

    # Кэш
    cache_key = f"scrapped_{config.hubs}_{parse_limit}_{hash(tuple(hubs_list))}"
    cache_path = config.get_cache_path(cache_key)
    articles_data = []

    if config.enable_cache and config.mode == OperationMode.NORMAL:
        cached = await load_cache(cache_path)
        if cached:
            articles_data = cached.get('articles', [])
            metrics.cache_hits += 1
            logger.info(f"Из кэша: {len(articles_data)} статей")
        else:
            metrics.cache_misses += 1
    else:
        metrics.cache_misses += 1

    if not articles_data:
        try:
            articles_data = await asyncio.wait_for(
                scraper._scrape_articles(parse_limit, hubs_list), timeout=config.timeout
            )
            if config.enable_cache:
                await save_cache(cache_path, {'articles': articles_data, 'timestamp': time.time()})
        except asyncio.TimeoutError:
            logger.error(f"Таймаут парсинга ({config.timeout}с)")
            metrics.status = PipelineStatus.FAILED
            return []
        except Exception as e:
            logger.error(f"Ошибка парсинга: {e}", exc_info=True)
            metrics.status = PipelineStatus.FAILED
            return []

    metrics.total_scraped = len(articles_data)
    if not articles_data:
        print("Статьи не найдены")
        metrics.status = PipelineStatus.COMPLETED
        return []

    # Валидация БД
    print(fmt_header("ФАЗА 2: ВАЛИДАЦИЯ БД"))
    metrics.status = PipelineStatus.VALIDATING
    try:
        async with AsyncSessionLocal() as session:
            repo = ArticleRepositoryImpl(session)
            existing = await repo.get_existing_urls([d['url'] for d in articles_data])
            new_data = [d for d in articles_data if d['url'] not in existing][:config.limit]

        print(fmt_row("Спарсено", len(articles_data)))
        print(fmt_row("В БД", len(existing)))
        print(fmt_row("Новых", len(new_data)))

        if not new_data:
            print("Нет новых статей")
            metrics.status = PipelineStatus.COMPLETED
        return new_data
    except Exception as e:
        logger.error(f"Ошибка валидации: {e}", exc_info=True)
        metrics.status = PipelineStatus.FAILED
        return []


# =========================================================================
# Проверка LLM провайдера
# =========================================================================

async def _check_llm_provider(config: PipelineConfig, metrics: PipelineMetrics) -> bool:
    """Проверить доступность LLM провайдера."""
    name = config.provider
    if not name:
        return False

    logger.info(f"[{metrics.correlation_id}] Проверка LLM: {name.upper()}")

    api_keys = {
        "openrouter": ("OPENROUTER_API_KEY", "sk-or-"),
        "groq": ("GROQ_API_KEY", "gsk_"),
        "google": ("GOOGLE_API_KEY", "AIza"),
    }

    if name in api_keys:
        env_var, _ = api_keys[name]
        key = os.getenv(env_var)
        if not key or "YOUR-KEY-HERE" in key or len(key) < 10:
            logger.error(f"[{metrics.correlation_id}] {env_var} не установлен или невалиден!")
            return False
        logger.info(f"[{metrics.correlation_id}] ✓ {env_var}: {key[:20]}...")

    if name == "ollama":
        try:
            import requests
            cfg = get_models_config(provider="ollama")
            resp = requests.get(f"{cfg.get_ollama_base_url()}/api/tags", timeout=10)
            if resp.status_code != 200:
                logger.error(f"Ollama недоступен: {resp.status_code}")
                return False
            models = resp.json().get("models", [])
            model_name = cfg.get_ollama_model()
            if not any(model_name in m.get("name", "") for m in models):
                logger.error(f"Модель {model_name} не найдена в Ollama")
                return False
            logger.info(f"✓ Ollama: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Ollama: {e}")
            return False

    # Тест провайдера
    try:
        mcfg = get_models_config(
            provider=name, strategy=config.strategy,
            enable_fallback=not config.no_fallback
        )
        test_config = mcfg.get_llm_config("classifier")
        from src.infrastructure.ai.llm_provider import LLMProviderFactory
        p = LLMProviderFactory.create(test_config)
        p.generate("Test", temperature=0.1, max_tokens=10)
        logger.info(f"[{metrics.correlation_id}] ✓ {name.upper()} OK")
        return True
    except Exception as e:
        logger.error(f"[{metrics.correlation_id}] Провайдер {name}: {e}")
        return False


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Pipeline обработки статей v3.2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Стандартный запуск (требует asyncpg)
  python %(prog)s 10

  # Конкретная статья (НЕ требует asyncpg!)
  python %(prog)s --url https://habr.com/ru/articles/123456/

  # Несколько статей
  python %(prog)s --url https://habr.com/ru/articles/111/,https://habr.com/ru/articles/222/

  # С публикацией
  python %(prog)s --url https://habr.com/ru/articles/123456/ --publish --provider groq
"""
    )
    parser.add_argument('limit', type=int, nargs='?', default=10)
    parser.add_argument('hubs', type=str, nargs='?', default="")
    parser.add_argument('--url', '-u', action='append', default=None,
                        help='URL статьи (можно несколько раз или через запятую)')
    parser.add_argument('--provider', '-p', choices=['groq', 'openrouter', 'google', 'ollama'])
    parser.add_argument('--no-fallback', action='store_true')
    parser.add_argument('--strategy', '-s',
                        choices=['cost_optimized', 'balanced', 'quality_focused', 'speed_focused'])
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--retry-delay', type=float, default=1.0)
    parser.add_argument('--retry-strategy', choices=['exponential', 'linear', 'fixed', 'adaptive'],
                        default='exponential')
    parser.add_argument('--max-concurrent', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--timeout', type=float, default=300.0)
    parser.add_argument('--mode', choices=['normal', 'resume', 'incremental', 'full'], default='normal')
    parser.add_argument('--enable-cache', action='store_true', default=True)
    parser.add_argument('--cache-dir', default='cache/pipeline')
    parser.add_argument('--enable-monitoring', action='store_true', default=True)
    parser.add_argument('--monitoring-interval', type=float, default=30.0)
    parser.add_argument('--save-failed-urls', action='store_true', default=True)
    parser.add_argument('--duplicate-check', action='store_true', default=True)
    parser.add_argument('--rate-limit', type=int)
    parser.add_argument('--health-check-interval', type=float, default=60.0)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--min-relevance', type=int, default=5)
    parser.add_argument('--publish', action='store_true', default=False)
    parser.add_argument('--telegraph', action='store_true', default=False)
    parser.add_argument('--telegram', action='store_true', default=False)
    parser.add_argument('--min-publish-score', type=int, default=7)

    args = parser.parse_args()
    urls = parse_url_args(args.url) if args.url else None

    try:
        asyncio.run(full_pipeline(
            limit=args.limit, hubs=args.hubs, verbose=args.verbose,
            min_relevance=args.min_relevance, debug=args.debug,
            provider=args.provider, strategy=args.strategy,
            no_fallback=args.no_fallback, max_retries=args.max_retries,
            retry_delay=args.retry_delay, max_concurrent=args.max_concurrent,
            batch_size=args.batch_size, timeout=args.timeout,
            mode=args.mode, retry_strategy=args.retry_strategy,
            enable_cache=args.enable_cache, cache_dir=args.cache_dir,
            enable_monitoring=args.enable_monitoring,
            monitoring_interval=args.monitoring_interval,
            enable_save_failed_urls=args.save_failed_urls,
            duplicate_check=args.duplicate_check, rate_limit=args.rate_limit,
            health_check_interval=args.health_check_interval,
            publish_telegraph=args.publish or args.telegraph,
            min_publish_score=args.min_publish_score if args.min_publish_score != 7 else args.min_relevance,
            publish_telegram=args.publish or args.telegram,
            urls=urls,
        ))
    except KeyboardInterrupt:
        print("\n⚠️  Прервано")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)


# docker compose exec postgres psql -U newsaggregator -d news_aggregator -c "DELETE FROM articles WHERE url LIKE '%1004288%'"
#
# docker compose exec api python run_full_pipeline.py --url https://habr.com/ru/news/1004288/ -p ollama --publish --verbose

# docker compose exec api python run_full_pipeline.py --url https://habr.com/ru/articles/1006098/ -p ollama --publish --min-relevance 1 --verbose