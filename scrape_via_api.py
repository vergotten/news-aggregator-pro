#!/usr/bin/env python3
"""
Парсер через API - работает БЕЗ прямого подключения к БД.
Использование: python scrape_via_api.py [limit] [hubs]
"""

import sys
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime


async def scrape_habr(limit: int = 10, hubs: str = ""):
    """Парсинг Habr и отправка через API."""
    
    print(f"\n🚀 Запуск Habr парсера через API")
    print(f"   Лимит: {limit}")
    print(f"   Хабы: {hubs if hubs else 'все'}\n")
    
    api_url = "http://localhost:8000/api/v1/articles/"
    habr_url = "https://habr.com/ru/articles/"
    
    stats = {'scraped': 0, 'saved': 0, 'duplicates': 0, 'errors': 0}
    
    # Парсим Habr
    async with aiohttp.ClientSession() as session:
        # Получаем HTML
        async with session.get(habr_url) as response:
            html = await response.text()
        
        soup = BeautifulSoup(html, 'html.parser')
        articles = soup.find_all('article', class_='tm-articles-list__item', limit=limit)
        
        stats['scraped'] = len(articles)
        
        # Отправляем каждую статью в API
        for article_card in articles:
            try:
                # Парсим данные
                title_elem = article_card.find('h2', class_='tm-title')
                if not title_elem:
                    continue
                
                title_link = title_elem.find('a')
                title = title_link.text.strip()
                url = "https://habr.com" + title_link['href']
                
                # Автор
                author_elem = article_card.find('a', class_='tm-user-info__username')
                author = author_elem.text.strip() if author_elem else None
                
                # Контент
                content_elem = article_card.find('div', class_='article-formatted-body')
                content = content_elem.text.strip()[:500] if content_elem else ""
                
                # Хабы
                hub_elems = article_card.find_all('a', class_='tm-publication-hub__link')
                article_hubs = [h.text.strip() for h in hub_elems]
                
                # Отправляем в API
                payload = {
                    "title": title,
                    "content": content,
                    "url": url,
                    "source": "habr",
                    "author": author,
                    "tags": article_hubs,
                    "hubs": article_hubs
                }
                
                async with session.post(api_url, json=payload) as resp:
                    if resp.status == 201:
                        stats['saved'] += 1
                        print(f"✓ {title[:50]}...")
                    elif resp.status == 400:
                        error = await resp.text()
                        if 'already exists' in error.lower():
                            stats['duplicates'] += 1
                            print(f"⊗ Дубликат: {title[:40]}...")
                        else:
                            stats['errors'] += 1
                            print(f"✗ Ошибка: {title[:40]}...")
                    else:
                        stats['errors'] += 1
                        print(f"✗ HTTP {resp.status}: {title[:40]}...")
                        
            except Exception as e:
                stats['errors'] += 1
                print(f"✗ Ошибка парсинга: {e}")
    
    # Итоги
    print(f"\n✅ Готово!")
    print(f"   Собрано: {stats['scraped']}")
    print(f"   Сохранено: {stats['saved']}")
    print(f"   Дубликатов: {stats['duplicates']}")
    print(f"   Ошибок: {stats['errors']}\n")


if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    hubs = sys.argv[2] if len(sys.argv) > 2 else ""
    
    asyncio.run(scrape_habr(limit, hubs))
