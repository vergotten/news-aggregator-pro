# -*- coding: utf-8 -*-
"""
SQLAlchemy модели — копия для Streamlit-контейнера.
Синхронизирована с src/infrastructure/persistence/models.py v3.1.
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Boolean, Float, JSON
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import declarative_base
import uuid

Base = declarative_base()


class ArticleModel(Base):
    __tablename__ = "articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    content = Column(Text)
    url = Column(String(2048), unique=True, index=True)
    source = Column(String(100), nullable=False, index=True)

    author = Column(String(255))
    published_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    status = Column(String(50), default="pending", index=True)
    embedding_status = Column(String(50), default="pending")
    vector_id = Column(String(255))

    is_news = Column(Boolean, default=False, index=True)
    relevance_score = Column(Float)
    relevance_reason = Column(Text)
    editorial_title = Column(String(500))
    editorial_teaser = Column(Text)
    editorial_rewritten = Column(Text)

    tags = Column(ARRAY(String), default=list)
    hubs = Column(ARRAY(String), default=list)
    images = Column(ARRAY(String), default=list)

    telegram_post_text = Column(Text)
    telegram_cover_image = Column(String(2048))
    telegraph_url = Column(String(2048))
    telegraph_content_html = Column(Text)

    seo_title = Column(String(200))
    seo_description = Column(Text)
    seo_slug = Column(String(500))
    seo_keywords = Column(ARRAY(String), default=list)
    seo_focus_keyword = Column(String(200))

    article_metadata = Column(JSON, default=dict)