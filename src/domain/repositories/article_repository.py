"""
Repository Interface: IArticleRepository

Порт (интерфейс) для работы с хранилищем статей.
Реализации (адаптеры) находятся в infrastructure layer.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set
from uuid import UUID

from src.domain.entities.article import Article
from src.domain.value_objects.article_status import ArticleStatus
from src.domain.value_objects.source_type import SourceType


class IArticleRepository(ABC):
    """
    Интерфейс репозитория статей.
    
    Следует Repository Pattern и является портом в Hexagonal Architecture.
    """
    
    @abstractmethod
    async def save(self, article: Article) -> Article:
        """
        Сохранить статью.
        
        Args:
            article: Статья для сохранения
            
        Returns:
            Сохранённая статья
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, article_id: UUID) -> Optional[Article]:
        """
        Найти статью по ID.
        
        Args:
            article_id: UUID статьи
            
        Returns:
            Статья или None
        """
        pass
    
    @abstractmethod
    async def find_by_url(self, url: str) -> Optional[Article]:
        """
        Найти статью по URL.
        
        Args:
            url: URL статьи
            
        Returns:
            Статья или None
        """
        pass
    
    @abstractmethod
    async def find_all(
        self,
        status: Optional[ArticleStatus] = None,
        source: Optional[SourceType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Article]:
        """
        Получить список статей с фильтрацией.
        
        Args:
            status: Фильтр по статусу
            source: Фильтр по источнику
            limit: Лимит записей
            offset: Смещение
            
        Returns:
            Список статей
        """
        pass
    
    @abstractmethod
    async def find_news(self, limit: int = 50, offset: int = 0) -> List[Article]:
        """
        Получить новостные статьи.
        
        Args:
            limit: Лимит записей
            offset: Смещение
            
        Returns:
            Список новостных статей
        """
        pass
    
    @abstractmethod
    async def count(
        self,
        status: Optional[ArticleStatus] = None,
        source: Optional[SourceType] = None
    ) -> int:
        """
        Подсчитать количество статей.
        
        Args:
            status: Фильтр по статусу
            source: Фильтр по источнику
            
        Returns:
            Количество статей
        """
        pass
    
    @abstractmethod
    async def delete(self, article_id: UUID) -> bool:
        """
        Удалить статью.
        
        Args:
            article_id: UUID статьи
            
        Returns:
            True если удалена
        """
        pass
    
    @abstractmethod
    async def exists_by_url(self, url: str) -> bool:
        """
        Проверить существование статьи по URL.
        
        Args:
            url: URL статьи
            
        Returns:
            True если существует
        """
        pass
    
    @abstractmethod
    async def get_existing_urls(self, urls: List[str]) -> Set[str]:
        """
        Массовая проверка существования статей по списку URLs.
        
        Эффективнее чем множественные вызовы exists_by_url().
        Используется для фильтрации уже обработанных статей при парсинге.
        
        Args:
            urls: Список URLs для проверки
            
        Returns:
            Множество URLs которые существуют в БД
            
        Example:
            urls = ['http://example.com/1', 'http://example.com/2']
            existing = await repo.get_existing_urls(urls)
            # → {'http://example.com/1'}  # только первый URL существует
        """
        pass
