"""
Command Handler для статей.
"""

from src.application.commands.create_article_command import CreateArticleCommand
from src.domain.entities.article import Article
from src.domain.repositories.article_repository import IArticleRepository
from src.shared.exceptions.domain_exceptions import DuplicateEntityError


class ArticleCommandHandler:
    """Handler для команд работы со статьями."""
    
    def __init__(self, repository: IArticleRepository):
        self.repository = repository
    
    async def handle_create_article(self, command: CreateArticleCommand) -> Article:
        """
        Обработка команды создания статьи.
        
        Args:
            command: Команда создания
            
        Returns:
            Созданная статья
            
        Raises:
            DuplicateEntityError: Если статья с таким URL уже существует
        """
        # Проверка на дубликат
        if await self.repository.exists_by_url(command.url):
            raise DuplicateEntityError(f"Article with URL {command.url} already exists")
        
        # Создание сущности
        article = Article(
            title=command.title,
            content=command.content,
            url=command.url,
            source=command.source,
            author=command.author,
            published_at=command.published_at,
            tags=command.tags,
            hubs=command.hubs,
            images=command.images,
            metadata=command.metadata
        )
        
        # Сохранение
        return await self.repository.save(article)
