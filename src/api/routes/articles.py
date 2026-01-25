"""
FastAPI Routes для статей.
"""

from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas.article_schemas import ArticleResponse, CreateArticleRequest
from src.application.services.article_service import ArticleService
from src.api.dependencies import get_article_service

router = APIRouter(prefix="/articles", tags=["articles"])


@router.post("/", response_model=ArticleResponse, status_code=201)
async def create_article(
    request: CreateArticleRequest,
    service: ArticleService = Depends(get_article_service)
):
    """Создать статью."""
    from src.application.commands.create_article_command import CreateArticleCommand
    
    command = CreateArticleCommand(
        title=request.title,
        content=request.content,
        url=request.url,
        source=request.source,
        author=request.author,
        published_at=request.published_at,
        tags=request.tags,
        hubs=request.hubs,
        images=request.images
    )
    
    article = await service.create_article(command)
    return ArticleResponse.from_entity(article)


@router.get("/{article_id}", response_model=ArticleResponse)
async def get_article(
    article_id: UUID,
    service: ArticleService = Depends(get_article_service)
):
    """Получить статью по ID."""
    article = await service.get_article(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return ArticleResponse.from_entity(article)


@router.get("/", response_model=List[ArticleResponse])
async def list_articles(
    limit: int = 100,
    offset: int = 0,
    is_news: bool = False,
    service: ArticleService = Depends(get_article_service)
):
    """Получить список статей."""
    articles = await service.list_articles(limit, offset, is_news)
    return [ArticleResponse.from_entity(a) for a in articles]
