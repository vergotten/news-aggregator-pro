# -*- coding: utf-8 -*-
"""
PostgreSQL Repository реализация.

v3.2: Добавлен маппинг cat_comment.
"""

from typing import List, Optional, Set
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.domain.entities.article import Article
from src.domain.repositories.article_repository import IArticleRepository
from src.domain.value_objects.article_status import ArticleStatus
from src.domain.value_objects.source_type import SourceType
from src.infrastructure.persistence.models import ArticleModel


class ArticleRepositoryImpl(IArticleRepository):

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, article: Article) -> Article:
        model = self._to_model(article)
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return self._to_entity(model)

    async def find_by_id(self, article_id: UUID) -> Optional[Article]:
        result = await self.session.execute(
            select(ArticleModel).where(ArticleModel.id == article_id)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def find_by_url(self, url: str) -> Optional[Article]:
        result = await self.session.execute(
            select(ArticleModel).where(ArticleModel.url == url)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def find_all(
        self,
        status: Optional[ArticleStatus] = None,
        source: Optional[SourceType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Article]:
        query = select(ArticleModel)
        if status:
            query = query.where(ArticleModel.status == status.value)
        if source:
            query = query.where(ArticleModel.source == source.value)
        query = query.order_by(ArticleModel.created_at.desc()).limit(limit).offset(offset)
        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def find_news(self, limit: int = 50, offset: int = 0) -> List[Article]:
        result = await self.session.execute(
            select(ArticleModel)
            .where(ArticleModel.is_news == True)
            .order_by(ArticleModel.published_at.desc())
            .limit(limit).offset(offset)
        )
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def count(
        self,
        status: Optional[ArticleStatus] = None,
        source: Optional[SourceType] = None
    ) -> int:
        query = select(func.count(ArticleModel.id))
        if status:
            query = query.where(ArticleModel.status == status.value)
        if source:
            query = query.where(ArticleModel.source == source.value)
        result = await self.session.execute(query)
        return result.scalar()

    async def delete(self, article_id: UUID) -> bool:
        model = await self.session.get(ArticleModel, article_id)
        if model:
            await self.session.delete(model)
            await self.session.commit()
            return True
        return False

    async def exists_by_url(self, url: str) -> bool:
        result = await self.session.execute(
            select(func.count(ArticleModel.id)).where(ArticleModel.url == url)
        )
        return result.scalar() > 0

    async def get_existing_urls(self, urls: List[str]) -> Set[str]:
        if not urls:
            return set()
        result = await self.session.execute(
            select(ArticleModel.url).where(ArticleModel.url.in_(urls))
        )
        return set(result.scalars().all())

    # =========================================================================
    # Маппинг Entity ↔ Model
    # =========================================================================

    def _to_model(self, entity: Article) -> ArticleModel:
        return ArticleModel(
            id=entity.id,
            title=entity.title,
            content=entity.content,
            url=entity.url,
            source=entity.source.value,
            author=entity.author,
            published_at=entity.published_at,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            status=entity.status.value,
            embedding_status=entity.embedding_status,
            vector_id=entity.vector_id,
            is_news=entity.is_news,
            relevance_score=entity.relevance_score,
            relevance_reason=entity.relevance_reason,
            editorial_title=entity.editorial_title,
            editorial_teaser=entity.editorial_teaser,
            editorial_rewritten=entity.editorial_rewritten,
            cat_comment_short=entity.cat_comment_short,
            cat_comment=entity.cat_comment,
            tags=entity.tags,
            hubs=entity.hubs,
            images=entity.images,
            telegram_post_text=entity.telegram_post_text,
            telegram_cover_image=entity.telegram_cover_image,
            telegraph_url=entity.telegraph_url,
            telegraph_content_html=entity.telegraph_content_html,
            seo_title=entity.seo_title,
            seo_description=entity.seo_description,
            seo_slug=entity.seo_slug,
            seo_keywords=entity.seo_keywords,
            seo_focus_keyword=entity.seo_focus_keyword,
            article_metadata=entity.metadata,
        )

    def _to_entity(self, model: ArticleModel) -> Article:
        return Article(
            id=model.id,
            title=model.title,
            content=model.content,
            url=model.url,
            source=SourceType(model.source),
            author=model.author,
            published_at=model.published_at,
            created_at=model.created_at,
            updated_at=model.updated_at,
            status=ArticleStatus(model.status),
            embedding_status=model.embedding_status,
            vector_id=model.vector_id,
            is_news=model.is_news,
            relevance_score=model.relevance_score,
            relevance_reason=model.relevance_reason,
            editorial_title=model.editorial_title,
            editorial_teaser=model.editorial_teaser,
            editorial_rewritten=model.editorial_rewritten,
            cat_comment_short=getattr(model, 'cat_comment_short', None),
            cat_comment=getattr(model, 'cat_comment', None),
            tags=model.tags or [],
            hubs=model.hubs or [],
            images=model.images or [],
            telegram_post_text=model.telegram_post_text,
            telegram_cover_image=model.telegram_cover_image,
            telegraph_url=model.telegraph_url,
            telegraph_content_html=model.telegraph_content_html,
            seo_title=getattr(model, 'seo_title', None),
            seo_description=getattr(model, 'seo_description', None),
            seo_slug=getattr(model, 'seo_slug', None),
            seo_keywords=getattr(model, 'seo_keywords', None) or [],
            seo_focus_keyword=getattr(model, 'seo_focus_keyword', None),
            metadata=model.article_metadata or {},
        )
