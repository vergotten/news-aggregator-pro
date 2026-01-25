"""
Domain Entity: Article

Представляет статью как доменную сущность со всей бизнес-логикой.
Следует принципам DDD (Domain-Driven Design).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4

from src.domain.value_objects.article_status import ArticleStatus
from src.domain.value_objects.source_type import SourceType
from src.shared.exceptions.domain_exceptions import DomainValidationError


@dataclass
class Article:
    """
    Доменная сущность статьи.
    
    Инвариант:
    - Статья всегда имеет уникальный ID
    - Заголовок не может быть пустым
    - URL должен быть валидным
    - Статус должен соответствовать бизнес-правилам
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    
    # Core attributes
    title: str = field(default="")
    content: str = field(default="")
    url: Optional[str] = None
    source: SourceType = SourceType.HABR
    
    # Metadata
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Processing
    status: ArticleStatus = ArticleStatus.PENDING
    embedding_status: str = "pending"
    vector_id: Optional[str] = None
    
    # Editorial
    is_news: bool = False
    relevance_score: Optional[float] = None
    relevance_reason: Optional[str] = None
    editorial_title: Optional[str] = None
    editorial_teaser: Optional[str] = None
    editorial_rewritten: Optional[str] = None
    
    # Collections
    tags: List[str] = field(default_factory=list)
    hubs: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    
    # Metadata JSON
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Валидация инвариантов после инициализации."""
        self.validate()
    
    def validate(self) -> None:
        """
        Проверка инвариантов сущности.
        
        Raises:
            DomainValidationError: Если инварианты нарушены
        """
        if not self.title or len(self.title.strip()) == 0:
            raise DomainValidationError("Article title cannot be empty")
        
        if len(self.title) > 500:
            raise DomainValidationError("Article title too long (max 500 chars)")
        
        if self.url and len(self.url) > 2048:
            raise DomainValidationError("Article URL too long (max 2048 chars)")
        
        if self.relevance_score is not None:
            if not 0 <= self.relevance_score <= 10:
                raise DomainValidationError("Relevance score must be between 0 and 10")
    
    def mark_as_processed(self) -> None:
        """Отметить статью как обработанную."""
        self.status = ArticleStatus.PROCESSED
        self.updated_at = datetime.utcnow()
    
    def mark_as_news(self, reason: str = "") -> None:
        """
        Отметить статью как новость.
        
        Args:
            reason: Причина классификации как новости
        """
        self.is_news = True
        self.relevance_reason = reason
        self.updated_at = datetime.utcnow()
    
    def set_relevance(self, score: float, reason: str) -> None:
        """
        Установить оценку релевантности.
        
        Args:
            score: Оценка от 0 до 10
            reason: Обоснование оценки
            
        Raises:
            DomainValidationError: Если score вне диапазона
        """
        if not 0 <= score <= 10:
            raise DomainValidationError(f"Invalid relevance score: {score}")
        
        self.relevance_score = score
        self.relevance_reason = reason
        self.updated_at = datetime.utcnow()
    
    def add_editorial_content(
        self,
        title: Optional[str] = None,
        teaser: Optional[str] = None,
        rewritten: Optional[str] = None
    ) -> None:
        """
        Добавить редакторский контент.
        
        Args:
            title: Отредактированный заголовок
            teaser: Краткое описание
            rewritten: Переписанный текст
        """
        if title:
            self.editorial_title = title
        if teaser:
            self.editorial_teaser = teaser
        if rewritten:
            self.editorial_rewritten = rewritten
        
        self.updated_at = datetime.utcnow()
    
    def set_vector_embedding(self, vector_id: str) -> None:
        """
        Установить ID векторного эмбеддинга.
        
        Args:
            vector_id: ID в векторной БД
        """
        self.vector_id = vector_id
        self.embedding_status = "completed"
        self.updated_at = datetime.utcnow()
    
    def is_duplicate_of(self, other: 'Article', threshold: float = 0.85) -> bool:
        """
        Проверка на дубликат (базовая проверка по заголовку).
        
        Полная проверка с векторным поиском должна быть в domain service.
        
        Args:
            other: Другая статья
            threshold: Порог схожести (не используется в базовой проверке)
            
        Returns:
            True если статьи дубликаты
        """
        # Простая проверка по заголовку
        return self.title.lower().strip() == other.title.lower().strip()
    
    def __eq__(self, other: object) -> bool:
        """Сравнение по ID."""
        if not isinstance(other, Article):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Хэш по ID."""
        return hash(self.id)
    
    def __repr__(self) -> str:
        return f"Article(id={self.id}, title='{self.title[:50]}...', source={self.source})"
