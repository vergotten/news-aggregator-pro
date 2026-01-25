"""
CQRS Query: GetArticleQuery

Запрос для получения статьи.
"""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class GetArticleQuery:
    """Запрос статьи по ID."""
    
    article_id: UUID


@dataclass(frozen=True)
class GetArticleByUrlQuery:
    """Запрос статьи по URL."""
    
    url: str


@dataclass(frozen=True)
class ListArticlesQuery:
    """Запрос списка статей."""
    
    limit: int = 100
    offset: int = 0
    is_news: bool = False
