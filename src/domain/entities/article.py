# -*- coding: utf-8 -*-
"""
Доменная сущность: Статья (Article)

v3.1: Полная версия с поддержкой:
- images: Массив URL изображений из статьи
- telegram_post_text: Готовый пост для Telegram
- telegram_cover_image: Обложка для Telegram
- telegraph_url: Ссылка на Telegraph
- telegraph_content_html: HTML для Telegraph
- seo_*: Поля SEO оптимизации
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4

from src.domain.value_objects.article_status import ArticleStatus
from src.domain.value_objects.source_type import SourceType
from src.shared.exceptions.domain_exceptions import DomainValidationError


@dataclass
class Article:
    """
    Доменная сущность статьи.

    Инварианты:
    - Статья всегда имеет уникальный ID
    - Заголовок не может быть пустым (max 500 символов)
    - URL не длиннее 2048 символов
    - Оценка релевантности: 0-10
    """

    # =========================================================================
    # Идентификация
    # =========================================================================
    id: UUID = field(default_factory=uuid4)

    # =========================================================================
    # Основные атрибуты
    # =========================================================================
    title: str = field(default="")
    content: str = field(default="")
    url: Optional[str] = None
    source: SourceType = SourceType.HABR

    # =========================================================================
    # Метаданные
    # =========================================================================
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # =========================================================================
    # Статус обработки
    # =========================================================================
    status: ArticleStatus = ArticleStatus.PENDING
    embedding_status: str = "pending"
    vector_id: Optional[str] = None

    # =========================================================================
    # Результаты AI обработки
    # =========================================================================
    is_news: bool = False
    relevance_score: Optional[float] = None
    relevance_reason: Optional[str] = None
    editorial_title: Optional[str] = None
    editorial_teaser: Optional[str] = None
    editorial_rewritten: Optional[str] = None

    # =========================================================================
    # Коллекции
    # =========================================================================
    tags: List[str] = field(default_factory=list)
    hubs: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)

    # =========================================================================
    # Telegram интеграция
    # =========================================================================
    telegram_post_text: Optional[str] = None
    telegram_cover_image: Optional[str] = None
    telegraph_url: Optional[str] = None
    telegraph_content_html: Optional[str] = None

    # =========================================================================
    # SEO оптимизация
    # =========================================================================
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    seo_slug: Optional[str] = None
    seo_keywords: List[str] = field(default_factory=list)
    seo_focus_keyword: Optional[str] = None

    # =========================================================================
    # JSON метаданные
    # =========================================================================
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Валидация инвариантов после инициализации."""
        self.validate()

    def validate(self) -> None:
        """
        Проверка инвариантов сущности.

        Исключения:
            DomainValidationError: Если инварианты нарушены
        """
        if not self.title or len(self.title.strip()) == 0:
            raise DomainValidationError("Article title cannot be empty")

        if len(self.title) > 500:
            raise DomainValidationError("Article title too long (max 500 chars)")

        if self.url and len(self.url) > 2048:
            raise DomainValidationError("Article URL too long (max 2048 chars)")

        if self.relevance_score is not None:
            if not 0 <= self.relevance_score <= 10:
                raise DomainValidationError("Relevance score must be between 0 and 10")

    # =========================================================================
    # Бизнес-логика
    # =========================================================================

    def mark_as_processed(self) -> None:
        """Отметить статью как обработанную."""
        self.status = ArticleStatus.PROCESSED
        self.updated_at = datetime.utcnow()

    def mark_as_news(self, reason: str = "") -> None:
        """Отметить статью как новость."""
        self.is_news = True
        self.relevance_reason = reason
        self.updated_at = datetime.utcnow()

    def set_relevance(self, score: float, reason: str) -> None:
        """
        Установить оценку релевантности.

        Аргументы:
            score: Оценка от 0 до 10
            reason: Обоснование оценки
        """
        if not 0 <= score <= 10:
            raise DomainValidationError(f"Invalid relevance score: {score}")

        self.relevance_score = score
        self.relevance_reason = reason
        self.updated_at = datetime.utcnow()

    def add_editorial_content(
        self,
        title: Optional[str] = None,
        teaser: Optional[str] = None,
        rewritten: Optional[str] = None
    ) -> None:
        """Добавить редакторский контент (заголовок, тизер, переписанный текст)."""
        if title:
            self.editorial_title = title
        if teaser:
            self.editorial_teaser = teaser
        if rewritten:
            self.editorial_rewritten = rewritten
        self.updated_at = datetime.utcnow()

    # =========================================================================
    # Telegram
    # =========================================================================

    def set_telegram_content(
        self,
        post_text: str,
        cover_image: Optional[str] = None,
        telegraph_url: Optional[str] = None,
        telegraph_html: Optional[str] = None
    ) -> None:
        """
        Установить Telegram контент.

        Аргументы:
            post_text: Готовый текст поста (с HTML разметкой)
            cover_image: URL обложки
            telegraph_url: URL статьи в Telegraph
            telegraph_html: HTML контент для Telegraph
        """
        self.telegram_post_text = post_text
        self.telegram_cover_image = cover_image
        self.telegraph_url = telegraph_url
        self.telegraph_content_html = telegraph_html
        self.updated_at = datetime.utcnow()

    def needs_telegraph(self) -> bool:
        """Нужен ли Telegraph (контент длиннее 5000 символов)."""
        return len(self.content) > 5000 if self.content else False

    # =========================================================================
    # SEO
    # =========================================================================

    def set_seo_data(
        self,
        title: str,
        description: str,
        slug: str,
        keywords: List[str],
        focus_keyword: Optional[str] = None
    ) -> None:
        """
        Установить SEO данные.

        Аргументы:
            title: SEO заголовок (50-60 символов)
            description: Meta description (150-160 символов)
            slug: URL-friendly slug
            keywords: Список ключевых слов
            focus_keyword: Главное ключевое слово
        """
        self.seo_title = title
        self.seo_description = description
        self.seo_slug = slug
        self.seo_keywords = keywords
        self.seo_focus_keyword = focus_keyword or (keywords[0] if keywords else None)
        self.updated_at = datetime.utcnow()

    # =========================================================================
    # Эмбеддинги и дубликаты
    # =========================================================================

    def set_vector_embedding(self, vector_id: str) -> None:
        """Установить ID векторного эмбеддинга."""
        self.vector_id = vector_id
        self.embedding_status = "completed"
        self.updated_at = datetime.utcnow()

    def is_duplicate_of(self, other: 'Article', threshold: float = 0.85) -> bool:
        """Проверка на дубликат (базовая — по заголовку)."""
        return self.title.lower().strip() == other.title.lower().strip()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Article):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Article(id={self.id}, title='{self.title[:50]}...', source={self.source})"