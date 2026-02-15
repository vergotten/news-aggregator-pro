#!/usr/bin/env python3
"""
Простой запуск парсера (без AI обработки) v4.1

Только сбор статей и сохранение в БД.
Для AI обработки используйте run_full_pipeline.py
"""

import asyncio
import sys
import argparse


async def run_habr_scraper(limit: int = 10, hubs: str = "", verbose: bool = False):
    """Запустить Habr парсер."""
    from src.scrapers.habr.scraper_service import HabrScraperService
    
    print(f"\n{'=' * 60}")
    print(f"🚀 HABR SCRAPER v4.1 (без AI обработки)")
    print(f"{'=' * 60}")
    print(f"  Лимит статей: {limit}")
    print(f"  Хабы: {hubs if hubs else 'все'}")
    print(f"{'=' * 60}\n")
    
    service = HabrScraperService()
    hubs_list = [h.strip() for h in hubs.split(',')] if hubs else []
    
    def progress_callback():
        if verbose:
            print(".", end="", flush=True)
    
    results = await service.scrape_and_save(
        limit=limit,
        hubs=hubs_list,
        progress_callback=progress_callback if verbose else None
    )
    
    if verbose:
        print()  # Новая строка после точек
    
    print(f"\n{'=' * 60}")
    print(f"✅ РЕЗУЛЬТАТЫ")
    print(f"{'=' * 60}")
    print(f"  Собрано:     {results['scraped']}")
    print(f"  Сохранено:   {results['saved']}")
    print(f"  Дубликатов:  {results['duplicates']}")
    print(f"  Ошибок:      {results['errors']}")
    print(f"{'=' * 60}")
    
    if results['saved'] > 0:
        print(f"\n💡 Для AI обработки выполните:")
        print(f"   python run_full_pipeline.py {results['saved']} --provider groq")
        print(f"   или")
        print(f"   python process_existing_articles.py --limit {results['saved']}")
    
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Habr парсер v4.1 (без AI обработки)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Собрать 50 статей
  python %(prog)s 50

  # С фильтром по хабам
  python %(prog)s 100 "python,machine-learning,devops"

  # С подробным выводом
  python %(prog)s 20 --verbose

Для AI обработки используйте:
  python run_full_pipeline.py 10 --provider groq
        """
    )
    
    parser.add_argument('limit', type=int, nargs='?', default=10,
                        help='Количество статей (default: 10)')
    parser.add_argument('hubs', type=str, nargs='?', default="",
                        help='Хабы через запятую')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Подробный вывод')
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_habr_scraper(args.limit, args.hubs, args.verbose))
    except KeyboardInterrupt:
        print("\n⚠️  Прервано")
        sys.exit(1)
