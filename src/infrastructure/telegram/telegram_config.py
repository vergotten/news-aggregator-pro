"""
Telegram Publishing Configuration.
"""

import random
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta


@dataclass
class TelegramPublishingConfig:
    """
    Конфигурация публикации в Telegram.
    """
    
    # Задержка между постами
    min_delay: int = 300  # 5 минут (минимум)
    max_delay: int = 1800  # 30 минут (максимум)
    
    # Jitter (случайное отклонение)
    jitter_minutes: int = 30  # +/- 30 минут
    
    # Расписание публикации
    publish_hours_start: int = 9   # Начало (9:00)
    publish_hours_end: int = 22    # Конец (22:00)
    
    # Дни недели (0 = Monday, 6 = Sunday)
    publish_days: list = None  # None = все дни
    
    # Максимум постов в день
    max_posts_per_day: int = 10
    
    # Пропускать выходные
    skip_weekends: bool = False
    
    def __post_init__(self):
        if self.publish_days is None:
            self.publish_days = list(range(7))  # Все дни
    
    def get_delay(self) -> int:
        """
        Получить задержку с jitter.
        
        Returns:
            Задержка в секундах
        """
        # Базовая задержка
        base_delay = random.randint(self.min_delay, self.max_delay)
        
        # Добавить jitter
        jitter_seconds = random.randint(
            -self.jitter_minutes * 60,
            self.jitter_minutes * 60
        )
        
        delay = base_delay + jitter_seconds
        
        # Минимум 60 секунд
        return max(60, delay)
    
    def should_publish_now(self) -> bool:
        """
        Проверить можно ли публиковать сейчас.
        
        Returns:
            True если можно публиковать
        """
        now = datetime.now()
        
        # Проверить день недели
        if now.weekday() not in self.publish_days:
            return False
        
        # Проверить выходные
        if self.skip_weekends and now.weekday() >= 5:
            return False
        
        # Проверить время
        if not (self.publish_hours_start <= now.hour < self.publish_hours_end):
            return False
        
        return True
    
    def get_next_publish_time(self) -> datetime:
        """
        Получить следующее время публикации.
        
        Returns:
            Datetime следующей публикации
        """
        now = datetime.now()
        
        # Начать с текущего времени + задержка
        next_time = now + timedelta(seconds=self.get_delay())
        
        # Если время вне расписания - сдвинуть на следующий день
        while not self._is_valid_time(next_time):
            next_time += timedelta(hours=1)
        
        return next_time
    
    def _is_valid_time(self, dt: datetime) -> bool:
        """Проверить что время валидное."""
        if dt.weekday() not in self.publish_days:
            return False
        
        if self.skip_weekends and dt.weekday() >= 5:
            return False
        
        if not (self.publish_hours_start <= dt.hour < self.publish_hours_end):
            return False
        
        return True


# Предустановленные конфигурации
TELEGRAM_PRESETS = {
    'aggressive': TelegramPublishingConfig(
        min_delay=60,      # 1 минута
        max_delay=300,     # 5 минут
        jitter_minutes=5,
        max_posts_per_day=50
    ),
    
    'normal': TelegramPublishingConfig(
        min_delay=300,     # 5 минут
        max_delay=1800,    # 30 минут
        jitter_minutes=30,
        max_posts_per_day=10
    ),
    
    'conservative': TelegramPublishingConfig(
        min_delay=1800,    # 30 минут
        max_delay=7200,    # 2 часа
        jitter_minutes=60,
        max_posts_per_day=5
    ),
    
    'business_hours': TelegramPublishingConfig(
        min_delay=600,     # 10 минут
        max_delay=3600,    # 1 час
        jitter_minutes=30,
        publish_hours_start=9,
        publish_hours_end=18,
        skip_weekends=True,
        max_posts_per_day=8
    )
}
