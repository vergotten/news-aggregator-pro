"""
CQRS Command: CreateArticleCommand

Команда для создания новой статьи.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

from src.domain.value_objects.source_type import SourceType


@dataclass(frozen=True)
class CreateArticleCommand:
    """
    Команда создания статьи.
    
    Иммутабельна (frozen=True) - следует принципу CQRS.
    """
    
    # Required
    title: str
    content: str
    url: str
    source: SourceType
    
    # Optional
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    tags: List[str] = None
    hubs: List[str] = None
    images: List[str] = None
    metadata: dict = None
    
    def __post_init__(self):
        """Установка значений по умолчанию для изменяемых типов."""
        if self.tags is None:
            object.__setattr__(self, 'tags', [])
        if self.hubs is None:
            object.__setattr__(self, 'hubs', [])
        if self.images is None:
            object.__setattr__(self, 'images', [])
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
