"""
Value Object: ArticleStatus

Статус обработки статьи.
"""

from enum import Enum


class ArticleStatus(str, Enum):
    """Статусы жизненного цикла статьи."""
    
    PENDING = "pending"              # Ожидает обработки
    PROCESSING = "processing"        # В процессе обработки
    PROCESSED = "processed"          # Обработана
    FAILED = "failed"                # Ошибка обработки
    DUPLICATE = "duplicate"          # Дубликат
    REJECTED = "rejected"            # Отклонена (не прошла фильтры)
    ARCHIVED = "archived"            # Архивирована
    
    def is_final(self) -> bool:
        """Проверка, является ли статус финальным."""
        return self in (
            self.PROCESSED,
            self.FAILED,
            self.DUPLICATE,
            self.REJECTED,
            self.ARCHIVED
        )
    
    def can_transition_to(self, new_status: 'ArticleStatus') -> bool:
        """
        Проверка возможности перехода в новый статус.
        
        Правила переходов:
        - PENDING -> PROCESSING, REJECTED
        - PROCESSING -> PROCESSED, FAILED, DUPLICATE
        - PROCESSED -> ARCHIVED
        - FAILED -> PROCESSING (retry)
        """
        transitions = {
            self.PENDING: [self.PROCESSING, self.REJECTED],
            self.PROCESSING: [self.PROCESSED, self.FAILED, self.DUPLICATE],
            self.PROCESSED: [self.ARCHIVED],
            self.FAILED: [self.PROCESSING],  # Retry
        }
        
        allowed = transitions.get(self, [])
        return new_status in allowed
