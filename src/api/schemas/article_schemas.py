"""
Pydantic schemas для API.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field

from src.domain.value_objects.source_type import SourceType
from src.domain.entities.article import Article


class CreateArticleRequest(BaseModel):
    """Запрос на создание статьи."""
    
    title: str = Field(..., min_length=1, max_length=500)
    content: str
    url: str = Field(..., max_length=2048)
    source: SourceType
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    tags: List[str] = []
    hubs: List[str] = []
    images: List[str] = []


class ArticleResponse(BaseModel):
    """Ответ со статьёй."""
    
    id: UUID
    title: str
    content: str
    url: str
    source: SourceType
    author: Optional[str]
    published_at: Optional[datetime]
    created_at: datetime
    status: str
    is_news: bool
    relevance_score: Optional[float]
    tags: List[str]
    hubs: List[str]
    
    @classmethod
    def from_entity(cls, entity: Article) -> "ArticleResponse":
        """Создать из entity."""
        return cls(
            id=entity.id,
            title=entity.title,
            content=entity.content,
            url=entity.url,
            source=entity.source,
            author=entity.author,
            published_at=entity.published_at,
            created_at=entity.created_at,
            status=entity.status.value,
            is_news=entity.is_news,
            relevance_score=entity.relevance_score,
            tags=entity.tags,
            hubs=entity.hubs
        )
    
    class Config:
        from_attributes = True
