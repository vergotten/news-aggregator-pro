"""
Telegram Publisher - публикация статей в Telegram.
"""

import asyncio
from typing import List, Optional
from telethon import TelegramClient
from telethon.tl.types import InputMediaPhotoExternal

from src.domain.entities.article import Article


class TelegramPublisher:
    """
    Сервис для публикации статей в Telegram канал/группу.
    """
    
    def __init__(
        self,
        api_id: str,
        api_hash: str,
        bot_token: Optional[str] = None,
        phone: Optional[str] = None
    ):
        """
        Инициализация Telegram клиента.
        
        Args:
            api_id: Telegram API ID (от my.telegram.org)
            api_hash: Telegram API Hash
            bot_token: Токен бота (опционально, для публикации через бота)
            phone: Номер телефона (если публикация от имени аккаунта)
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.bot_token = bot_token
        self.phone = phone
        
        self.client = None
    
    async def connect(self):
        """Подключиться к Telegram."""
        if self.bot_token:
            # Публикация через бота
            self.client = TelegramClient(
                'bot_session',
                self.api_id,
                self.api_hash
            )
            await self.client.start(bot_token=self.bot_token)
        elif self.phone:
            # Публикация от имени аккаунта
            self.client = TelegramClient(
                'user_session',
                self.api_id,
                self.api_hash
            )
            await self.client.start(phone=self.phone)
        else:
            raise ValueError("Нужен bot_token или phone")
    
    async def disconnect(self):
        """Отключиться от Telegram."""
        if self.client:
            await self.client.disconnect()
    
    async def publish_article(
        self,
        article: Article,
        channel: str,
        include_images: bool = True,
        min_relevance: float = 7.0
    ) -> bool:
        """
        Опубликовать статью в канал/группу.
        
        Args:
            article: Статья для публикации
            channel: ID или username канала (например: @my_channel или -1001234567890)
            include_images: Включать ли изображения
            min_relevance: Минимальная релевантность для публикации
            
        Returns:
            True если опубликовано, False если пропущено
        """
        # Фильтр по релевантности
        if article.relevance_score and article.relevance_score < min_relevance:
            print(f"⏭️  Пропущено: низкая релевантность ({article.relevance_score}/10)")
            return False
        
        # Подготовить текст
        message = self._format_message(article)
        
        try:
            # Публикация с изображениями
            if include_images and article.images:
                await self._publish_with_images(channel, message, article.images)
            else:
                # Публикация только текста
                await self.client.send_message(
                    channel,
                    message,
                    parse_mode='html',
                    link_preview=False
                )
            
            print(f"✅ Опубликовано: {article.editorial_title or article.title}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка публикации: {e}")
            return False
    
    async def _publish_with_images(
        self,
        channel: str,
        message: str,
        images: List[dict]
    ):
        """
        Опубликовать с изображениями.
        
        Args:
            channel: Канал
            message: Текст сообщения
            images: Список изображений
        """
        # Telegram поддерживает до 10 изображений в одном посте
        image_urls = [img['url'] for img in images[:10] if img.get('url')]
        
        if not image_urls:
            # Если URL'ов нет - просто текст
            await self.client.send_message(
                channel,
                message,
                parse_mode='html',
                link_preview=False
            )
            return
        
        if len(image_urls) == 1:
            # Одно изображение
            await self.client.send_message(
                channel,
                message,
                file=image_urls[0],
                parse_mode='html'
            )
        else:
            # Несколько изображений (медиа-группа)
            media = [InputMediaPhotoExternal(url) for url in image_urls]
            await self.client.send_file(
                channel,
                file=media,
                caption=message,
                parse_mode='html'
            )
    
    def _format_message(self, article: Article) -> str:
        """
        Форматировать сообщение для Telegram.
        
        Args:
            article: Статья
            
        Returns:
            Отформатированное сообщение
        """
        # Заголовок
        title = article.editorial_title or article.title
        
        # Тизер (краткое описание)
        teaser = article.editorial_teaser or ""
        
        # Текст (нормализованный или оригинальный)
        content = article.editorial_rewritten or article.content or ""
        
        # Ограничить длину для ТГ (4096 символов максимум)
        max_content_length = 2000  # Оставим место для заголовка и ссылок
        if len(content) > max_content_length:
            content = content[:max_content_length]
            # Найти конец последнего предложения
            last_period = content.rfind('.')
            if last_period > max_content_length * 0.8:
                content = content[:last_period + 1]
            else:
                content += "..."
        
        # Теги
        tags = ""
        if article.tags:
            tags = " ".join([f"#{tag.replace(' ', '_')}" for tag in article.tags[:5]])
        
        # Собрать сообщение
        parts = []
        
        # Заголовок (жирный)
        parts.append(f"<b>{title}</b>")
        
        # Тизер (если есть)
        if teaser:
            parts.append(f"\n<i>{teaser}</i>")
        
        # Основной текст
        if content:
            parts.append(f"\n\n{content}")
        
        # Ссылка на оригинал
        if article.url:
            parts.append(f"\n\n🔗 <a href='{article.url}'>Читать полностью</a>")
        
        # Теги
        if tags:
            parts.append(f"\n\n{tags}")
        
        # Метаданные (опционально)
        meta = []
        if article.relevance_score:
            meta.append(f"📊 {article.relevance_score}/10")
        if article.is_news:
            meta.append("📰 Новость")
        
        if meta:
            parts.append(f"\n\n{' | '.join(meta)}")
        
        return "".join(parts)
    
    async def publish_batch(
        self,
        articles: List[Article],
        channel: str,
        config: Optional['TelegramPublishingConfig'] = None,
        **kwargs
    ) -> dict:
        """
        Опубликовать несколько статей с умной задержкой.
        
        Args:
            articles: Список статей
            channel: Канал
            config: Конфигурация публикации (опционально)
            **kwargs: Дополнительные параметры для publish_article
            
        Returns:
            Статистика: {published, skipped, errors, next_publish_time}
        """
        from src.infrastructure.telegram.telegram_config import TelegramPublishingConfig
        
        if config is None:
            config = TelegramPublishingConfig()
        
        stats = {
            'published': 0,
            'skipped': 0,
            'errors': 0,
            'delays': []
        }
        
        for i, article in enumerate(articles, 1):
            # Проверить можно ли публиковать
            if not config.should_publish_now():
                print(f"⏸️  Пауза: вне расписания публикации")
                stats['skipped'] += len(articles) - i + 1
                break
            
            # Проверить лимит постов в день
            if stats['published'] >= config.max_posts_per_day:
                print(f"⏸️  Достигнут лимит: {config.max_posts_per_day} постов/день")
                stats['skipped'] += len(articles) - i + 1
                break
            
            print(f"\n[{i}/{len(articles)}] {article.title}")
            
            try:
                if await self.publish_article(article, channel, **kwargs):
                    stats['published'] += 1
                else:
                    stats['skipped'] += 1
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                stats['errors'] += 1
            
            # Задержка перед следующим постом (кроме последнего)
            if i < len(articles):
                delay = config.get_delay()
                stats['delays'].append(delay)
                
                minutes = delay // 60
                print(f"⏳ Ждём {minutes} мин. (с jitter)...")
                await asyncio.sleep(delay)
        
        # Следующее время публикации
        stats['next_publish_time'] = config.get_next_publish_time()
        
        return stats
