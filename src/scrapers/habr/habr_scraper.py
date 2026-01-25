"""
Habr Scraper с интеграцией в архитектуру.
"""

import scrapy
from datetime import datetime
from typing import Optional, Dict, Any

from src.application.commands.create_article_command import CreateArticleCommand
from src.domain.value_objects.source_type import SourceType


class HabrArticleSpider(scrapy.Spider):
    """Spider для Habr."""
    
    name = "habr_articles"
    allowed_domains = ["habr.com"]
    start_urls = ["https://habr.com/ru/articles/"]
    
    def parse(self, response):
        """Парсинг страницы со статьями."""
        # Простой пример - нужно доработать селекторы
        for article in response.css("article.tm-articles-list__item"):
            yield {
                "title": article.css("h2.tm-title::text").get(),
                "url": response.urljoin(article.css("a.tm-title__link::attr(href)").get()),
                "teaser": article.css("div.article-formatted-body::text").get(),
            }


def create_article_from_scraped(data: Dict[str, Any]) -> CreateArticleCommand:
    """Создать команду из scraped данных."""
    return CreateArticleCommand(
        title=data["title"],
        content=data.get("content", ""),
        url=data["url"],
        source=SourceType.HABR,
        author=data.get("author"),
        published_at=data.get("published_at"),
        tags=data.get("tags", []),
        hubs=data.get("hubs", [])
    )
