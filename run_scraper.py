#!/usr/bin/env python3
"""
Простой запуск парсера (синхронный wrapper).
"""

import asyncio
import sys


async def run_habr_scraper(limit: int = 10, hubs: str = ""):
    """Запустить Habr парсер."""
    from src.scrapers.habr.scraper_service import HabrScraperService
    
    print(f"\n🚀 Запуск Habr парсера")
    print(f"   Лимит: {limit}")
    print(f"   Хабы: {hubs if hubs else 'все'}\n")
    
    service = HabrScraperService()
    hubs_list = [h.strip() for h in hubs.split(',')] if hubs else []
    
    results = await service.scrape_and_save(
        limit=limit,
        hubs=hubs_list
    )
    
    print(f"\n✅ Готово!")
    print(f"   Собрано: {results['scraped']}")
    print(f"   Сохранено: {results['saved']}")
    print(f"   Дубликатов: {results['duplicates']}")
    print(f"   Ошибок: {results['errors']}\n")


if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    hubs = sys.argv[2] if len(sys.argv) > 2 else ""
    
    asyncio.run(run_habr_scraper(limit, hubs))
