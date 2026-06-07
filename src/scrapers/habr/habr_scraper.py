# -*- coding: utf-8 -*-
"""
Habr Scraper - FIXED версия с улучшенным парсингом изображений.

Исправления:
- Правильный парсинг изображений из <figure>, <img> тегов
- Обработка data-src, srcset атрибутов
- Извлечение высококачественных версий изображений
- Поддержка всех форматов изображений на Habr
"""

import scrapy
from datetime import datetime
from typing import Optional, Dict, Any
from urllib.parse import urljoin

from src.application.commands.create_article_command import CreateArticleCommand
from src.domain.value_objects.source_type import SourceType


class HabrArticleSpider(scrapy.Spider):
    """
    Spider для парсинга статей с Habr.

    Особенности:
    - Парсит полный контент статьи
    - Извлекает все изображения (включая figure, img)
    - Обрабатывает lazy-loading изображения
    - Извлекает высококачественные версии
    """

    name = "habr_articles"
    allowed_domains = ["habr.com"]

    def __init__(self, limit=10, hubs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.hubs = hubs or []
        self.articles_parsed = 0

        # Базовый URL зависит от фильтров
        if self.hubs:
            self.start_urls = [
                f"https://habr.com/ru/flows/develop/articles/"
            ]
        else:
            self.start_urls = ["https://habr.com/ru/articles/"]

    def parse(self, response):
        """Парсинг страницы со списком статей."""

        # Находим все карточки статей
        article_cards = response.css('article.tm-articles-list__item')

        for card in article_cards:
            if self.articles_parsed >= self.limit:
                return

            # Извлекаем базовую информацию
            title_elem = card.css('h2.tm-title a')
            if not title_elem:
                continue

            article_url = response.urljoin(title_elem.css('::attr(href)').get())

            # Переходим на страницу статьи для полного парсинга
            yield scrapy.Request(
                article_url,
                callback=self.parse_article,
                meta={'dont_redirect': True}
            )

            self.articles_parsed += 1

    def parse_article(self, response):
        """Парсинг полной статьи."""

        try:
            # Заголовок
            title = response.css('h1.tm-title span::text').get()
            if not title:
                title = response.css('h1.tm-article-snippet__title span::text').get()

            # Автор
            author = response.css('a.tm-user-info__username::text').get()

            # Дата публикации
            time_elem = response.css('time::attr(datetime)').get()
            published_at = datetime.fromisoformat(time_elem) if time_elem else datetime.utcnow()

            # Хабы
            hubs = response.css('a.tm-publication-hub__link span::text').getall()

            # Теги (берём из хабов + дополнительные если есть)
            tags = hubs.copy()

            # Контент статьи
            article_body = response.css('div.tm-article-body')
            if not article_body:
                article_body = response.css('div.article-formatted-body')

            # Извлекаем текст
            content = self._extract_content(article_body)

            # ✨ ГЛАВНОЕ: Извлекаем ВСЕ изображения правильно
            images = self._extract_images(article_body, response)

            yield {
                'title': title,
                'content': content,
                'url': response.url,
                'author': author,
                'published_at': published_at,
                'tags': tags,
                'hubs': hubs,
                'images': images,  # Массив URL строк
                'source': SourceType.HABR.value
            }

        except Exception as e:
            self.logger.error(f"Error parsing article {response.url}: {e}")

    def _extract_content(self, article_body) -> str:
        """
        Извлечение текстового контента статьи.

        Сохраняет структуру с параграфами, заголовками, списками.
        """
        if not article_body:
            return ""

        content_parts = []

        # Проходим по всем элементам сохраняя структуру
        for elem in article_body.css('*'):
            tag = elem.root.tag
            text = elem.css('::text').get()

            if not text:
                continue

            text = text.strip()
            if not text:
                continue

            # Заголовки
            if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                content_parts.append(f"\n\n{text}\n")

            # Параграфы
            elif tag == 'p':
                content_parts.append(f"{text}\n\n")

            # Списки
            elif tag == 'li':
                content_parts.append(f"• {text}\n")

            # Код
            elif tag in ['code', 'pre']:
                content_parts.append(f"\n```\n{text}\n```\n\n")

            # Остальной текст
            elif tag not in ['script', 'style', 'img', 'svg']:
                content_parts.append(text + " ")

        return ''.join(content_parts).strip()

    def _extract_images(self, article_body, response) -> list[str]:
        """
        Извлечение ВСЕХ изображений из статьи.

        Обрабатывает:
        - <img> теги с src, data-src
        - <figure> с изображениями
        - srcset для высококачественных версий
        - Lazy-loading изображения

        Args:
            article_body: Selector контента статьи
            response: Response объект для urljoin

        Returns:
            Список URL изображений (высококачественные версии)
        """
        if not article_body:
            return []

        images = []
        seen_urls = set()  # Для дедупликации

        # 1. Изображения в <figure> тегах (основной формат Habr)
        for figure in article_body.css('figure'):
            img = figure.css('img')
            if img:
                img_url = self._get_best_image_url(img, response)
                if img_url and img_url not in seen_urls:
                    images.append(img_url)
                    seen_urls.add(img_url)

        # 2. Прямые <img> теги (не в figure)
        for img in article_body.css('img'):
            # Пропускаем если уже в figure
            if img.css('::attr(class)').get() and 'full-width' in img.css('::attr(class)').get():
                img_url = self._get_best_image_url(img, response)
                if img_url and img_url not in seen_urls:
                    images.append(img_url)
                    seen_urls.add(img_url)

        # 3. Изображения внутри ссылок
        for link_img in article_body.css('a img'):
            img_url = self._get_best_image_url(link_img, response)
            if img_url and img_url not in seen_urls:
                images.append(img_url)
                seen_urls.add(img_url)

        # 4. Lazy-loading изображения (data-src)
        for lazy_img in article_body.css('img[data-src]'):
            img_url = self._get_best_image_url(lazy_img, response)
            if img_url and img_url not in seen_urls:
                images.append(img_url)
                seen_urls.add(img_url)

        return images

    def _get_best_image_url(self, img_selector, response) -> Optional[str]:
        """
        Получить лучший URL изображения (наивысшее качество).

        Приоритет:
        1. data-src (часто содержит оригинал)
        2. srcset (выбираем максимальное разрешение)
        3. src (стандартный атрибут)

        Args:
            img_selector: Selector изображения
            response: Response для urljoin

        Returns:
            URL изображения или None
        """
        # 1. data-src (Habr использует для lazy loading оригиналов)
        data_src = img_selector.css('::attr(data-src)').get()
        if data_src:
            return self._normalize_image_url(data_src, response)

        # 2. srcset (может содержать версии в разных разрешениях)
        srcset = img_selector.css('::attr(srcset)').get()
        if srcset:
            # srcset формат: "url1 1x, url2 2x, url3 3x"
            # Берём максимальное разрешение (последний обычно)
            urls = srcset.split(',')
            if urls:
                # Парсим последний URL (обычно наивысшее разрешение)
                last_url = urls[-1].strip().split()[0]
                return self._normalize_image_url(last_url, response)

        # 3. src (стандартный)
        src = img_selector.css('::attr(src)').get()
        if src:
            return self._normalize_image_url(src, response)

        return None

    def _normalize_image_url(self, url: str, response) -> str:
        """
        Нормализация URL изображения.

        - Относительные URL → абсолютные
        - Protocol-relative URL (//example.com) → https://example.com
        - Убираем query parameters если не нужны
        """
        if not url:
            return ""

        url = url.strip()

        # Protocol-relative URL
        if url.startswith('//'):
            url = 'https:' + url

        # Относительный URL
        elif url.startswith('/'):
            url = urljoin('https://habr.com', url)

        # Полный URL через response.urljoin для всех случаев
        url = response.urljoin(url)

        return url


def create_article_from_scraped(data: Dict[str, Any]) -> CreateArticleCommand:
    """
    Создать команду для сохранения статьи из scraped данных.

    Args:
        data: Данные из spider

    Returns:
        CreateArticleCommand готовая к выполнению
    """
    return CreateArticleCommand(
        title=data["title"],
        content=data.get("content", ""),
        url=data["url"],
        source=SourceType.HABR,
        author=data.get("author"),
        published_at=data.get("published_at"),
        tags=data.get("tags", []),
        hubs=data.get("hubs", []),
        images=data.get("images", [])  # ✅ Массив URL строк
    )
