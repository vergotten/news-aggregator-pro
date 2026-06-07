# -*- coding: utf-8 -*-
"""
SQLAlchemy модели — инфраструктурный слой.

v3.1: Полная синхронизация с доменной сущностью Article.
Добавлены:
- telegram_post_text, telegram_cover_image, telegraph_url, telegraph_content_html
- seo_title, seo_description, seo_slug, seo_keywords, seo_focus_keyword
- images (массив URL изображений)
- article_metadata (JSON)
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Boolean, Float, JSON
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()


class ArticleModel(Base):
    """
    SQLAlchemy модель статьи.

    v3.1: Полная синхронизация с Article entity.
    Все поля из доменной сущности представлены в таблице.

    Маппинг особенностей:
    - entity.metadata ↔ model.article_metadata (избегаем конфликт с SQLAlchemy)
    """

    __tablename__ = "articles"

    # =========================================================================
    # Основные поля
    # =========================================================================
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    content = Column(Text)
    url = Column(String(2048), unique=True, index=True)
    source = Column(String(100), nullable=False, index=True)

    # =========================================================================
    # Метаданные
    # =========================================================================
    author = Column(String(255))
    published_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # =========================================================================
    # Статус обработки
    # =========================================================================
    status = Column(String(50), default="pending", index=True)
    embedding_status = Column(String(50), default="pending")
    vector_id = Column(String(255))

    # =========================================================================
    # Результаты AI обработки
    # =========================================================================
    is_news = Column(Boolean, default=False, index=True)
    relevance_score = Column(Float)
    relevance_reason = Column(Text)
    editorial_title = Column(String(500))
    editorial_teaser = Column(Text)
    editorial_rewritten = Column(Text)

    # =========================================================================
    # Коллекции (массивы)
    # =========================================================================
    tags = Column(ARRAY(String), default=list)
    hubs = Column(ARRAY(String), default=list)
    images = Column(ARRAY(String), default=list, comment="URL изображений из статьи")

    # =========================================================================
    # Telegram интеграция
    # =========================================================================
    telegram_post_text = Column(
        Text,
        comment="Готовый HTML пост для Telegram"
    )
    telegram_cover_image = Column(
        String(2048),
        comment="URL обложки для Telegram поста"
    )
    telegraph_url = Column(
        String(2048),
        comment="URL полной статьи в Telegraph (для длинных статей)"
    )
    telegraph_content_html = Column(
        Text,
        comment="HTML контент для публикации в Telegraph"
    )

    # =========================================================================
    # SEO оптимизация
    # =========================================================================
    seo_title = Column(
        String(200),
        comment="SEO заголовок (50-60 символов)"
    )
    seo_description = Column(
        Text,
        comment="Meta description (150-160 символов)"
    )
    seo_slug = Column(
        String(500),
        comment="URL-friendly slug"
    )
    seo_keywords = Column(
        ARRAY(String),
        default=list,
        comment="Массив ключевых слов для SEO"
    )
    seo_focus_keyword = Column(
        String(200),
        comment="Главное ключевое слово"
    )

    # =========================================================================
    # JSON метаданные
    # =========================================================================
    # ВАЖНО: Названо article_metadata (не metadata) чтобы не конфликтовать
    # с SQLAlchemy.metadata. В доменной сущности это поле называется metadata.
    article_metadata = Column(
        JSON,
        default=dict,
        comment="Дополнительные метаданные в формате JSON"
    )

    def __repr__(self):
        return f"<ArticleModel(id={self.id}, title='{self.title[:50]}...')>"