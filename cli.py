#!/usr/bin/env python3
"""
CLI для запуска парсеров.

Использование:
    python cli.py scrape habr --limit 10
    python cli.py scrape telegram --channels "@tech_news"
"""

import asyncio
import click
from rich.console import Console
from rich.progress import Progress

console = Console()


@click.group()
def cli():
    """News Aggregator CLI."""
    pass


@cli.group()
def scrape():
    """Команды для запуска парсеров."""
    pass


@scrape.command()
@click.option('--limit', default=10, help='Количество статей')
@click.option('--hubs', default='', help='Хабы через запятую')
async def habr(limit: int, hubs: str):
    """
    Запустить парсер Habr.
    
    Примеры:
        python cli.py scrape habr --limit 20
        python cli.py scrape habr --limit 50 --hubs "python,devops"
    """
    console.print(f"\n🚀 [bold green]Запуск Habr парсера[/bold green]")
    console.print(f"Лимит: {limit}")
    console.print(f"Хабы: {hubs if hubs else 'все'}\n")
    
    from src.scrapers.habr.scraper_service import HabrScraperService
    
    service = HabrScraperService()
    hubs_list = [h.strip() for h in hubs.split(',')] if hubs else []
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Парсинг статей...", total=limit)
        
        results = await service.scrape_and_save(
            limit=limit,
            hubs=hubs_list,
            progress_callback=lambda: progress.update(task, advance=1)
        )
    
    console.print(f"\n✅ [bold green]Готово![/bold green]")
    console.print(f"Собрано: {results['scraped']}")
    console.print(f"Сохранено: {results['saved']}")
    console.print(f"Дубликатов: {results['duplicates']}")
    console.print(f"Ошибок: {results['errors']}\n")


@scrape.command()
@click.option('--channels', required=True, help='Каналы через запятую')
@click.option('--limit', default=100, help='Количество сообщений')
async def telegram(channels: str, limit: int):
    """
    Запустить парсер Telegram.
    
    Примеры:
        python cli.py scrape telegram --channels "@tech_news,@python_news" --limit 50
    """
    console.print(f"\n🚀 [bold green]Запуск Telegram парсера[/bold green]")
    console.print(f"Каналы: {channels}")
    console.print(f"Лимит: {limit}\n")
    
    console.print("[yellow]⚠️  Telegram парсер в разработке[/yellow]\n")


if __name__ == '__main__':
    cli()
