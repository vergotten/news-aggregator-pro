"""
FastAPI Dependencies для DI.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.config.database import get_db_session
from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl
from src.application.handlers.article_command_handler import ArticleCommandHandler
from src.application.services.article_service import ArticleService


async def get_article_repository(
    session: AsyncSession = Depends(get_db_session)
) -> ArticleRepositoryImpl:
    """DI для repository."""
    return ArticleRepositoryImpl(session)


async def get_article_service(
    repository: ArticleRepositoryImpl = Depends(get_article_repository)
) -> ArticleService:
    """DI для service."""
    command_handler = ArticleCommandHandler(repository)
    return ArticleService(repository, command_handler)
