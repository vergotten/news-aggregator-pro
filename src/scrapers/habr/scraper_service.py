# -*- coding: utf-8 -*-
"""
Habr Scraper Service v3.1

Изменения v3.1:
- Интеграция с SkiplistService
- Пропуск URL из skiplist при парсинге
- Логирование пропущенных URL
"""

import asyncio
from typing import List, Dict, Optional, Callable
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
import re
import logging

from src.application.commands.create_article_command import CreateArticleCommand
from src.application.services.article_service import ArticleService
from src.domain.value_objects.source_type import SourceType
from src.infrastructure.config.database import AsyncSessionLocal
from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl
from src.application.handlers.article_command_handler import ArticleCommandHandler

# Skiplist
from src.infrastructure.skiplist import get_skiplist_service

logger = logging.getLogger(__name__)


class HabrScraperService:
    """
    Сервис парсинга Habr v3.1

    Интегрирован со SkiplistService для пропуска проблемных URL.
    """

    BASE_URL = "https://habr.com"

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    def __init__(self):
        """Инициализация с skiplist."""
        self.skiplist = get_skiplist_service()
        logger.info(f"[Scraper] Initialized, skiplist: {len(self.skiplist.list_urls())} URLs")

    async def scrape_and_save(
        self,
        limit: int = 10,
        hubs: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, int]:
        """
        Спарсить статьи и сохранить.

        Returns:
            Статистика: {scraped, saved, duplicates, errors, skipped}
        """
        stats = {
            'scraped': 0,
            'saved': 0,
            'duplicates': 0,
            'errors': 0,
            'skipped': 0  # Пропущено из skiplist
        }

        articles_data = await self._scrape_articles(limit, hubs, stats)
        stats['scraped'] = len(articles_data)

        async with AsyncSessionLocal() as session:
            repository = ArticleRepositoryImpl(session)
            command_handler = ArticleCommandHandler(repository)
            service = ArticleService(repository, command_handler)

            for article_data in articles_data:
                try:
                    command = CreateArticleCommand(
                        title=article_data['title'],
                        content=article_data['content'],
                        url=article_data['url'],
                        source=SourceType.HABR,
                        author=article_data.get('author'),
                        published_at=article_data.get('published_at'),
                        tags=article_data.get('tags', []),
                        hubs=article_data.get('hubs', []),
                        images=article_data.get('images', [])
                    )

                    await service.create_article(command)
                    stats['saved'] += 1

                except Exception as e:
                    if 'already exists' in str(e).lower():
                        stats['duplicates'] += 1
                    else:
                        stats['errors'] += 1
                        logger.error(f"[Scraper] Save error: {article_data.get('title', 'Unknown')}: {e}")

                if progress_callback:
                    progress_callback()

        return stats

    async def _scrape_articles(
        self,
        limit: int,
        hubs: Optional[List[str]] = None,
        stats: Optional[Dict] = None
    ) -> List[Dict]:
        """Парсинг статей с проверкой skiplist."""
        articles = []

        if hubs:
            url = f"{self.BASE_URL}/ru/flows/develop/articles/"
        else:
            url = f"{self.BASE_URL}/ru/articles/"

        async with aiohttp.ClientSession(headers=self.HEADERS) as session:
            async with session.get(url) as response:
                html = await response.text()

        soup = BeautifulSoup(html, 'html.parser')
        article_cards = soup.find_all('article', class_='tm-articles-list__item', limit=limit * 2)

        for card in article_cards:
            if len(articles) >= limit:
                break

            try:
                title_elem = card.find('h2', class_='tm-title')
                if not title_elem:
                    continue

                title_link = title_elem.find('a')
                if not title_link:
                    continue

                article_url = self.BASE_URL + title_link['href']

                # =========================================================
                # ПРОВЕРКА SKIPLIST
                # =========================================================
                if self.skiplist.should_skip(article_url):
                    reason = self.skiplist.get_reason(article_url)
                    logger.info(f"[Scraper] SKIPPED: {article_url[:50]}... ({reason.value if reason else 'unknown'})")
                    if stats:
                        stats['skipped'] = stats.get('skipped', 0) + 1
                    continue

                # Парсим статью
                article_data = await self._parse_full_article(article_url)
                if article_data:
                    articles.append(article_data)

            except Exception as e:
                logger.error(f"[Scraper] Card parse error: {e}")

        return articles

    async def _parse_full_article(self, url: str) -> Optional[Dict]:
        """Парсинг полной статьи."""
        try:
            async with aiohttp.ClientSession(headers=self.HEADERS) as session:
                async with session.get(url) as response:
                    html = await response.text()

            soup = BeautifulSoup(html, 'html.parser')

            # Заголовок
            title_elem = soup.find('h1', class_='tm-title')
            if not title_elem:
                title_elem = soup.find('h1', class_='tm-article-snippet__title')

            title = title_elem.find('span').text.strip() if title_elem else "Untitled"

            # Автор
            author_elem = soup.find('a', class_='tm-user-info__username')
            author = author_elem.text.strip() if author_elem else None

            # Дата
            time_elem = soup.find('time')
            published_at = None
            if time_elem and time_elem.get('datetime'):
                try:
                    published_at = datetime.fromisoformat(time_elem['datetime'])
                except:
                    published_at = datetime.utcnow()
            else:
                published_at = datetime.utcnow()

            # Хабы
            hubs = []
            hub_elems = soup.find_all('a', class_='tm-publication-hub__link')
            for hub_elem in hub_elems:
                hub_span = hub_elem.find('span')
                if hub_span:
                    hubs.append(hub_span.text.strip())

            tags = hubs.copy()

            # Контент
            article_body = soup.find('div', class_='tm-article-body')
            if not article_body:
                article_body = soup.find('div', class_='article-formatted-body')

            if not article_body:
                logger.warning(f"[Scraper] No content: {url}")
                return None

            content = self._extract_content_text(article_body)
            images = self._extract_all_images(article_body)

            print(f"[OK] {title[:50]}... | {len(content)} chars | {len(images)} images")

            return {
                'title': title,
                'content': content,
                'url': url,
                'author': author,
                'published_at': published_at,
                'tags': tags,
                'hubs': hubs,
                'images': images
            }

        except Exception as e:
            logger.error(f"[Scraper] Parse error {url}: {e}")
            return None

    def _extract_content_text(self, article_body) -> str:
        """Извлечение текста."""
        if not article_body:
            return ""

        for tag in article_body.find_all(['script', 'style']):
            tag.decompose()

        text = article_body.get_text(separator='\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _extract_all_images(self, article_body) -> List[str]:
        """Извлечение изображений."""
        if not article_body:
            return []

        images = []
        seen_urls = set()

        # Figure tags
        for figure in article_body.find_all('figure'):
            img = figure.find('img')
            if img:
                img_url = self._get_best_image_url(img)
                if img_url and img_url not in seen_urls:
                    images.append(img_url)
                    seen_urls.add(img_url)

        # Direct img tags
        for img in article_body.find_all('img'):
            if img.find_parent('figure'):
                continue
            img_url = self._get_best_image_url(img)
            if img_url and img_url not in seen_urls:
                images.append(img_url)
                seen_urls.add(img_url)

        # Images in links
        for link in article_body.find_all('a'):
            img = link.find('img')
            if img:
                href = link.get('href', '')
                if any(ext in href.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                    img_url = self._normalize_image_url(href)
                    if img_url and img_url not in seen_urls:
                        images.append(img_url)
                        seen_urls.add(img_url)
                else:
                    img_url = self._get_best_image_url(img)
                    if img_url and img_url not in seen_urls:
                        images.append(img_url)
                        seen_urls.add(img_url)

        return images

    def _get_best_image_url(self, img_tag) -> Optional[str]:
        """Лучший URL изображения."""
        if not img_tag:
            return None

        # data-src
        data_src = img_tag.get('data-src')
        if data_src:
            return self._normalize_image_url(data_src)

        # srcset
        srcset = img_tag.get('srcset')
        if srcset:
            parts = srcset.split(',')
            if parts:
                last_part = parts[-1].strip()
                url = last_part.split()[0]
                return self._normalize_image_url(url)

        # src
        src = img_tag.get('src')
        if src:
            return self._normalize_image_url(src)

        return None

    def _normalize_image_url(self, url: str) -> Optional[str]:
        """Нормализация URL."""
        if not url:
            return None

        url = url.strip()

        if url.startswith('//'):
            url = 'https:' + url
        elif url.startswith('/'):
            url = self.BASE_URL + url

        if not any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']):
            if not url.startswith('http'):
                return None

        return url

    async def scrape_single_article(self, url: str) -> Optional[Dict]:
        """Парсинг одной статьи."""
        return await self._parse_full_article(url)

    def get_skiplist_stats(self) -> dict:
        """Статистика skiplist."""
        return self.skiplist.get_stats()
