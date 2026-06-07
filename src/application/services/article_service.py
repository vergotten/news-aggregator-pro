"""
Application Service для управления статьями.
"""

from typing import List, Optional
from uuid import UUID

from src.application.commands.create_article_command import CreateArticleCommand
from src.application.handlers.article_command_handler import ArticleCommandHandler
from src.domain.entities.article import Article
from src.domain.repositories.article_repository import IArticleRepository


class ArticleService:
    """
    Application Service для статей.
    
    Координирует работу между handlers и domain services.
    """
    
    def __init__(
        self,
        repository: IArticleRepository,
        command_handler: ArticleCommandHandler
    ):
        self.repository = repository
        self.command_handler = command_handler
    
    async def create_article(self, command: CreateArticleCommand) -> Article:
        """Создать статью."""
        return await self.command_handler.handle_create_article(command)
    
    async def get_article(self, article_id: UUID) -> Optional[Article]:
        """Получить статью по ID."""
        return await self.repository.find_by_id(article_id)
    
    async def list_articles(
        self,
        limit: int = 100,
        offset: int = 0,
        is_news: bool = False
    ) -> List[Article]:
        """Получить список статей."""
        if is_news:
            return await self.repository.find_news(limit, offset)
        return await self.repository.find_all(limit=limit, offset=offset)
