"""
Value Object: SourceType

Тип источника контента.
"""

from enum import Enum


class SourceType(str, Enum):
    """Поддерживаемые источники контента."""
    
    HABR = "habr"
    TELEGRAM = "telegram"
    REDDIT = "reddit"
    MEDIUM = "medium"
    RSS = "rss"
    CUSTOM = "custom"
    
    @property
    def display_name(self) -> str:
        """Человекочитаемое имя."""
        names = {
            self.HABR: "Habr",
            self.TELEGRAM: "Telegram",
            self.REDDIT: "Reddit",
            self.MEDIUM: "Medium",
            self.RSS: "RSS Feed",
            self.CUSTOM: "Custom Source"
        }
        return names[self]
    
    @property
    def requires_auth(self) -> bool:
        """Требуется ли аутентификация."""
        return self in (self.TELEGRAM, self.REDDIT)
