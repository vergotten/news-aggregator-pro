"""
Habr Scraper Service - интеграция с архитектурой.
FIXED: images теперь возвращает массив строк (URLs), а не словарей.
"""

import asyncio
from typing import List, Dict, Optional, Callable
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup

from src.application.commands.create_article_command import CreateArticleCommand
from src.application.services.article_service import ArticleService
from src.domain.value_objects.source_type import SourceType
from src.infrastructure.config.database import AsyncSessionLocal
from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl
from src.application.handlers.article_command_handler import ArticleCommandHandler


class HabrScraperService:
    """
    Сервис для парсинга Habr.

    Интегрирован с Hexagonal Architecture.

    FIXED v2.0: images теперь массив строк (URLs), не словарей.
    """

    BASE_URL = "https://habr.com"

    async def scrape_and_save(
        self,
        limit: int = 10,
        hubs: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, int]:
        """
        Спарсить статьи и сохранить в БД.

        Args:
            limit: Количество статей
            hubs: Список хабов для фильтрации
            progress_callback: Callback для прогресса

        Returns:
            Статистика: {scraped, saved, duplicates, errors}
        """
        stats = {
            'scraped': 0,
            'saved': 0,
            'duplicates': 0,
            'errors': 0
        }

        # Парсим статьи
        articles_data = await self._scrape_articles(limit, hubs)
        stats['scraped'] = len(articles_data)

        # Сохраняем через Application Service
        async with AsyncSessionLocal() as session:
            repository = ArticleRepositoryImpl(session)
            command_handler = ArticleCommandHandler(repository)
            service = ArticleService(repository, command_handler)

            for article_data in articles_data:
                try:
                    # Создаём команду
                    command = CreateArticleCommand(
                        title=article_data['title'],
                        content=article_data['content'],
                        url=article_data['url'],
                        source=SourceType.HABR,
                        author=article_data.get('author'),
                        published_at=article_data.get('published_at'),
                        tags=article_data.get('tags', []),
                        hubs=article_data.get('hubs', []),
                        images=article_data.get('images', [])  # Теперь это массив строк!
                    )

                    # Сохраняем через service
                    await service.create_article(command)
                    stats['saved'] += 1

                except Exception as e:
                    if 'already exists' in str(e).lower():
                        stats['duplicates'] += 1
                    else:
                        stats['errors'] += 1
                        print(f"Ошибка: {e}")

                if progress_callback:
                    progress_callback()

        return stats

    async def _scrape_articles(
        self,
        limit: int,
        hubs: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Парсинг статей с Habr.

        Args:
            limit: Количество статей
            hubs: Фильтр по хабам

        Returns:
            Список статей
        """
        articles = []

        # URL для парсинга
        if hubs:
            url = f"{self.BASE_URL}/ru/flows/develop/articles/"
        else:
            url = f"{self.BASE_URL}/ru/articles/"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()

        soup = BeautifulSoup(html, 'html.parser')

        # Парсим статьи
        article_cards = soup.find_all('article', class_='tm-articles-list__item', limit=limit)

        for card in article_cards:
            try:
                # Базовые данные из карточки
                article_data = self._parse_article_card(card)
                if not article_data:
                    continue

                # Парсим полный текст и изображения
                if article_data.get('_needs_full_parse'):
                    full_data = await self._parse_full_article(article_data['url'])
                    if full_data:
                        article_data['content'] = full_data['content']
                        article_data['images'] = full_data['images']

                    del article_data['_needs_full_parse']

                articles.append(article_data)

            except Exception as e:
                print(f"Ошибка парсинга: {e}")

        return articles

    def _parse_article_card(self, card) -> Optional[Dict]:
        """
        Парсинг одной карточки статьи.

        Args:
            card: BeautifulSoup element

        Returns:
            Данные статьи
        """
        try:
            # Заголовок
            title_elem = card.find('h2', class_='tm-title')
            if not title_elem:
                return None

            title_link = title_elem.find('a')
            title = title_link.text.strip()
            url = self.BASE_URL + title_link['href']

            # Автор
            author_elem = card.find('a', class_='tm-user-info__username')
            author = author_elem.text.strip() if author_elem else None

            # Превью текста (для списка)
            content_elem = card.find('div', class_='article-formatted-body')
            preview = content_elem.text.strip()[:500] if content_elem else ""

            # Хабы
            hubs = []
            hub_elems = card.find_all('a', class_='tm-publication-hub__link')
            for hub_elem in hub_elems:
                hubs.append(hub_elem.text.strip())

            # Теги (часто совпадают с хабами на Habr)
            tags = hubs.copy()

            return {
                'title': title,
                'content': preview,  # Пока preview, полный текст парсим отдельно
                'url': url,
                'author': author,
                'published_at': datetime.utcnow(),
                'tags': tags,
                'hubs': hubs,
                '_needs_full_parse': True  # Флаг что нужен полный парсинг
            }

        except Exception as e:
            print(f"Ошибка парсинга карточки: {e}")
            return None

    async def _parse_full_article(self, url: str) -> Optional[Dict]:
        """
        Парсинг ПОЛНОЙ статьи со страницы.

        Args:
            url: URL статьи

        Returns:
            Полный текст и изображения (только URLs!)
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    html = await response.text()

            soup = BeautifulSoup(html, 'html.parser')

            # Найти контент статьи
            article_body = soup.find('div', class_='tm-article-body')
            if not article_body:
                article_body = soup.find('div', class_='article-formatted-body')

            if not article_body:
                return None

            # Полный текст
            full_text = article_body.get_text(separator='\n', strip=True)

            # Изображения - ТОЛЬКО URLs (массив строк для PostgreSQL VARCHAR[])
            images = []
            for img in article_body.find_all('img'):
                img_url = img.get('src') or img.get('data-src')
                if img_url:
                    # Относительные URL в абсолютные
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    elif img_url.startswith('/'):
                        img_url = self.BASE_URL + img_url

                    # ✅ ИСПРАВЛЕНО: Добавляем только URL (строку), не словарь!
                    images.append(img_url)

            return {
                'content': full_text,
                'images': images  # ✅ Теперь ['url1', 'url2'] вместо [{'url': ...}]
            }

        except Exception as e:
            print(f"Ошибка парсинга полной статьи {url}: {e}")
            return None