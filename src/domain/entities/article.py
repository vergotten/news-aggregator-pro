# -*- coding: utf-8 -*-
"""
Доменная сущность: Статья (Article)

v3.2: Добавлено поле cat_comment для НейроКота
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
    # НейроКот комментарий
    # =========================================================================
    cat_comment_short: Optional[str] = None  # Для Telegram (1-2 предложения)
    cat_comment: Optional[str] = None        # Для статьи (3-5 предложений)

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
        self.validate()

    def validate(self) -> None:
        if not self.title or len(self.title.strip()) == 0:
            raise DomainValidationError("Article title cannot be empty")
        if len(self.title) > 500:
            raise DomainValidationError("Article title too long (max 500 chars)")
        if self.url and len(self.url) > 2048:
            raise DomainValidationError("Article URL too long (max 2048 chars)")
        if self.relevance_score is not None:
            if not 0 <= self.relevance_score <= 10:
                raise DomainValidationError("Relevance score must be between 0 and 10")

    def mark_as_processed(self) -> None:
        self.status = ArticleStatus.PROCESSED
        self.updated_at = datetime.utcnow()

    def mark_as_news(self, reason: str = "") -> None:
        self.is_news = True
        self.relevance_reason = reason
        self.updated_at = datetime.utcnow()

    def set_relevance(self, score: float, reason: str) -> None:
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
        if title:
            self.editorial_title = title
        if teaser:
            self.editorial_teaser = teaser
        if rewritten:
            self.editorial_rewritten = rewritten
        self.updated_at = datetime.utcnow()

    def set_cat_comment(self, comment: str) -> None:
        """Установить длинный комментарий НейроКота (для статьи)."""
        self.cat_comment = comment
        self.updated_at = datetime.utcnow()

    def set_cat_comment_short(self, comment: str) -> None:
        """Установить короткий комментарий НейроКота (для Telegram)."""
        self.cat_comment_short = comment
        self.updated_at = datetime.utcnow()

    def set_telegram_content(
        self,
        post_text: str,
        cover_image: Optional[str] = None,
        telegraph_url: Optional[str] = None,
        telegraph_html: Optional[str] = None
    ) -> None:
        self.telegram_post_text = post_text
        self.telegram_cover_image = cover_image
        self.telegraph_url = telegraph_url
        self.telegraph_content_html = telegraph_html
        self.updated_at = datetime.utcnow()

    def needs_telegraph(self) -> bool:
        return len(self.content) > 5000 if self.content else False

    def set_seo_data(
        self,
        title: str,
        description: str,
        slug: str,
        keywords: List[str],
        focus_keyword: Optional[str] = None
    ) -> None:
        self.seo_title = title
        self.seo_description = description
        self.seo_slug = slug
        self.seo_keywords = keywords
        self.seo_focus_keyword = focus_keyword or (keywords[0] if keywords else None)
        self.updated_at = datetime.utcnow()

    def set_vector_embedding(self, vector_id: str) -> None:
        self.vector_id = vector_id
        self.embedding_status = "completed"
        self.updated_at = datetime.utcnow()

    def is_duplicate_of(self, other: 'Article', threshold: float = 0.85) -> bool:
        return self.title.lower().strip() == other.title.lower().strip()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Article):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Article(id={self.id}, title='{self.title[:50]}...', source={self.source})"