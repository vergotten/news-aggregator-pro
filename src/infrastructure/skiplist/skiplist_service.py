# -*- coding: utf-8 -*-
"""
Skiplist Service — управление списком проблемных URL

Хранит URL которые:
- Слишком длинные для обработки (context limit)
- Вызвали ошибки при обработке
- Требуют ручной проверки

Расположение: src/infrastructure/skiplist/
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)


class SkipReason(Enum):
    """Причины добавления в skiplist."""
    CONTEXT_TOO_LONG = "context_too_long"
    PROCESSING_ERROR = "processing_error"
    EMPTY_RESPONSE = "empty_response"
    RATE_LIMITED = "rate_limited"
    MANUAL = "manual"
    DUPLICATE = "duplicate"


@dataclass
class SkipEntry:
    """Запись в skiplist."""
    url: str
    reason: SkipReason
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())
    content_length: int = 0
    error_message: str = ""
    retry_count: int = 0
    expires_at: Optional[str] = None  # Для временных блокировок (rate limit)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['reason'] = self.reason.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'SkipEntry':
        data['reason'] = SkipReason(data['reason'])
        return cls(**data)

    def is_expired(self) -> bool:
        """Проверить истекла ли временная блокировка."""
        if not self.expires_at:
            return False
        try:
            expires = datetime.fromisoformat(self.expires_at)
            return datetime.now() > expires
        except:
            return False


class SkiplistService:
    """
    Сервис управления skiplist.

    Использование:
        skiplist = SkiplistService()

        # Проверить
        if skiplist.should_skip(url):
            logger.info("Skipping...")

        # Добавить
        skiplist.add(url, SkipReason.CONTEXT_TOO_LONG, content_length=121000)

        # Временная блокировка (rate limit)
        skiplist.add_temporary(url, SkipReason.RATE_LIMITED, minutes=5)

        # Удалить (для retry)
        skiplist.remove(url)

        # Очистить
        skiplist.clear()
    """

    # Путь по умолчанию — в data директории проекта
    DEFAULT_PATH = Path("data/skiplist.json")

    def __init__(self, filepath: Optional[Path] = None):
        """
        Инициализация.

        Args:
            filepath: Путь к файлу (по умолчанию data/skiplist.json)
        """
        self.filepath = filepath or self.DEFAULT_PATH
        self._entries: Dict[str, SkipEntry] = {}
        self._ensure_directory()
        self._load()
        logger.info(f"[Skiplist] Loaded {len(self._entries)} entries from {self.filepath}")

    def _ensure_directory(self):
        """Создать директорию если не существует."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        """Загрузить из файла."""
        if not self.filepath.exists():
            return

        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for url, entry_data in data.get('entries', {}).items():
                try:
                    entry = SkipEntry.from_dict(entry_data)
                    # Пропускаем истекшие записи
                    if not entry.is_expired():
                        self._entries[url] = entry
                except Exception as e:
                    logger.warning(f"[Skiplist] Skip invalid entry {url}: {e}")

        except Exception as e:
            logger.warning(f"[Skiplist] Load error: {e}")

    def _save(self):
        """Сохранить в файл."""
        try:
            data = {
                'updated_at': datetime.now().isoformat(),
                'count': len(self._entries),
                'entries': {url: entry.to_dict() for url, entry in self._entries.items()}
            }

            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"[Skiplist] Save error: {e}")

    def should_skip(self, url: str) -> bool:
        """
        Проверить нужно ли пропустить URL.

        Автоматически удаляет истекшие записи.
        """
        if url not in self._entries:
            return False

        entry = self._entries[url]

        # Проверяем истечение срока
        if entry.is_expired():
            del self._entries[url]
            self._save()
            logger.info(f"[Skiplist] Expired entry removed: {url[:50]}...")
            return False

        return True

    def get_reason(self, url: str) -> Optional[SkipReason]:
        """Получить причину пропуска."""
        entry = self._entries.get(url)
        return entry.reason if entry else None

    def get_entry(self, url: str) -> Optional[SkipEntry]:
        """Получить полную запись."""
        return self._entries.get(url)

    def add(
        self,
        url: str,
        reason: SkipReason,
        content_length: int = 0,
        error_message: str = ""
    ) -> None:
        """
        Добавить URL в skiplist (постоянно).

        Args:
            url: URL статьи
            reason: Причина
            content_length: Длина контента
            error_message: Текст ошибки
        """
        if url in self._entries:
            # Увеличиваем счетчик retry
            self._entries[url].retry_count += 1
            self._entries[url].reason = reason
            if error_message:
                self._entries[url].error_message = error_message
        else:
            self._entries[url] = SkipEntry(
                url=url,
                reason=reason,
                content_length=content_length,
                error_message=error_message
            )

        self._save()
        logger.info(f"[Skiplist] Added: {url[:60]}... ({reason.value})")

    def add_temporary(
        self,
        url: str,
        reason: SkipReason,
        minutes: int = 5,
        error_message: str = ""
    ) -> None:
        """
        Добавить временную блокировку (для rate limit).

        Args:
            url: URL
            reason: Причина
            minutes: На сколько минут
            error_message: Текст ошибки
        """
        expires = datetime.now() + timedelta(minutes=minutes)

        self._entries[url] = SkipEntry(
            url=url,
            reason=reason,
            error_message=error_message,
            expires_at=expires.isoformat()
        )

        self._save()
        logger.info(f"[Skiplist] Temporary block ({minutes}m): {url[:60]}...")

    def remove(self, url: str) -> bool:
        """
        Удалить URL из skiplist.

        Returns:
            True если был удалён
        """
        if url in self._entries:
            del self._entries[url]
            self._save()
            logger.info(f"[Skiplist] Removed: {url[:60]}...")
            return True
        return False

    def clear(self) -> int:
        """
        Очистить весь skiplist.

        Returns:
            Количество удалённых записей
        """
        count = len(self._entries)
        self._entries.clear()
        self._save()
        logger.info(f"[Skiplist] Cleared {count} entries")
        return count

    def clear_by_reason(self, reason: SkipReason) -> int:
        """Очистить записи с определённой причиной."""
        to_remove = [url for url, e in self._entries.items() if e.reason == reason]
        for url in to_remove:
            del self._entries[url]
        self._save()
        logger.info(f"[Skiplist] Cleared {len(to_remove)} entries with reason {reason.value}")
        return len(to_remove)

    def cleanup_expired(self) -> int:
        """Удалить все истекшие записи."""
        expired = [url for url, e in self._entries.items() if e.is_expired()]
        for url in expired:
            del self._entries[url]
        if expired:
            self._save()
            logger.info(f"[Skiplist] Cleaned up {len(expired)} expired entries")
        return len(expired)

    def list_all(self) -> List[SkipEntry]:
        """Получить все записи."""
        return list(self._entries.values())

    def list_urls(self) -> Set[str]:
        """Получить все URL."""
        return set(self._entries.keys())

    def get_stats(self) -> dict:
        """Статистика skiplist."""
        by_reason = {}
        for entry in self._entries.values():
            reason = entry.reason.value
            by_reason[reason] = by_reason.get(reason, 0) + 1

        total_size = sum(e.content_length for e in self._entries.values())

        return {
            'total': len(self._entries),
            'by_reason': by_reason,
            'total_content_size': total_size,
            'filepath': str(self.filepath)
        }

    def print_stats(self):
        """Вывести статистику в консоль."""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("SKIPLIST STATISTICS")
        print("=" * 50)
        print(f"Total entries: {stats['total']}")
        print(f"File: {stats['filepath']}")
        print(f"\nBy reason:")
        for reason, count in stats['by_reason'].items():
            print(f"  {reason}: {count}")
        print("=" * 50 + "\n")


# Глобальный экземпляр
_instance: Optional[SkiplistService] = None


def get_skiplist_service() -> SkiplistService:
    """Получить глобальный экземпляр SkiplistService."""
    global _instance
    if _instance is None:
        _instance = SkiplistService()
    return _instance


def reset_skiplist_service():
    """Сбросить глобальный экземпляр (для тестов)."""
    global _instance
    _instance = None
