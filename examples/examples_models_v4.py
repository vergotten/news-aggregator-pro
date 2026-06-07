#!/usr/bin/env python3
"""
Примеры использования Models Config v4.0

Демонстрирует новые возможности:
- Provider-first approach
- Автоматический fallback
- Гибкие стратегии
"""

import os
import sys
from pathlib import Path

# Добавить путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models_config import get_models_config, reset_models_config
from src.infrastructure.ai.smart_llm_provider import create_smart_provider


def example_1_simple_usage():
    """Пример 1: Простое использование через env переменные."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 1: Простое использование")
    print("=" * 80)
    
    # Установить провайдера через env
    os.environ["LLM_PROVIDER"] = "groq"
    os.environ["LLM_STRATEGY"] = "balanced"
    
    # Получить конфиг
    config = get_models_config()
    config.print_config()
    
    # Создать провайдера для агента
    provider = create_smart_provider("summarizer")
    
    # Использовать
    result = provider.generate(
        "Summarize: Python is a high-level programming language..."
    )
    
    if result:
        print(f"\n✅ Результат: {result[:100]}...")
    
    # Статистика
    provider.print_stats()


def example_2_fallback_chain():
    """Пример 2: Автоматический fallback."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 2: Автоматический fallback")
    print("=" * 80)
    
    reset_models_config()
    
    # Groq → OpenRouter → Google → Ollama
    os.environ["LLM_PROVIDER"] = "groq"
    os.environ["LLM_FALLBACK_CHAIN"] = "speed_first"
    
    config = get_models_config()
    
    print(f"\n🔄 Fallback цепочка: {' → '.join(config.get_fallback_providers())}")
    
    # Если Groq упадёт (rate limit) - автоматически переключится
    provider = create_smart_provider("classifier")
    
    # Симуляция нескольких запросов
    for i in range(5):
        result = provider.generate(f"Classify article {i}: ...")
        if result:
            print(f"  [{i+1}] ✅ Success")
        else:
            print(f"  [{i+1}] ❌ Failed")
    
    provider.print_stats()


def example_3_cost_optimization():
    """Пример 3: Оптимизация затрат."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 3: Оптимизация затрат (только FREE модели)")
    print("=" * 80)
    
    reset_models_config()
    
    os.environ["LLM_PROVIDER"] = "openrouter"
    os.environ["LLM_STRATEGY"] = "cost_optimized"  # Только бесплатные
    
    config = get_models_config()
    config.print_config()
    
    # Все агенты будут использовать только бесплатные модели
    for agent in ["classifier", "summarizer", "rewriter"]:
        llm_config = config.get_llm_config(agent)
        print(f"  {agent:20} → {llm_config.model}")


def example_4_quality_focus():
    """Пример 4: Фокус на качестве."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 4: Максимальное качество")
    print("=" * 80)
    
    reset_models_config()
    
    os.environ["LLM_PROVIDER"] = "openrouter"
    os.environ["LLM_STRATEGY"] = "quality_focused"
    os.environ["LLM_FALLBACK_CHAIN"] = "quality_first"
    
    config = get_models_config()
    config.print_config()
    
    # Сложные задачи получат премиум модели
    for agent in ["summarizer", "rewriter", "style_normalizer"]:
        llm_config = config.get_llm_config(agent)
        print(f"  {agent:20} → {llm_config.model}")


def example_5_programmatic():
    """Пример 5: Программное управление."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 5: Программное управление")
    print("=" * 80)
    
    reset_models_config()
    
    # Создать конфиг программно
    config = get_models_config(
        provider="groq",
        strategy="balanced",
        fallback_chain="speed_first"
    )
    
    print(f"\n📋 Provider: {config.provider_name}")
    print(f"📊 Strategy: {config.strategy}")
    print(f"🔄 Fallback: {' → '.join(config.get_fallback_providers())}")
    
    # Переопределить провайдера для конкретного агента
    custom_config = config.get_llm_config(
        "summarizer",
        provider_override="google"  # Использовать Google вместо Groq
    )
    
    print(f"\n🔧 Custom config для summarizer:")
    print(f"  Provider: {custom_config.provider.value}")
    print(f"  Model: {custom_config.model}")


def example_6_docker_compose():
    """Пример 6: Использование в Docker Compose."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 6: Docker Compose")
    print("=" * 80)
    
    print("""
В docker-compose.yml:

services:
  api:
    environment:
      # Простой вариант
      LLM_PROVIDER: groq
      GROQ_API_KEY: ${GROQ_API_KEY}
      
      # Или более детальный
      LLM_PROVIDER: openrouter
      LLM_STRATEGY: balanced
      LLM_FALLBACK_CHAIN: quality_first
      OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}
      GOOGLE_API_KEY: ${GOOGLE_API_KEY}

Запуск:

  # Groq приоритет
  LLM_PROVIDER=groq docker-compose exec api python run_pipeline.py
  
  # OpenRouter приоритет
  LLM_PROVIDER=openrouter docker-compose exec api python run_pipeline.py
  
  # Только бесплатные модели
  LLM_PROVIDER=groq LLM_STRATEGY=cost_optimized docker-compose exec api python run_pipeline.py
  
  # Максимальное качество
  LLM_PROVIDER=openrouter LLM_STRATEGY=quality_focused docker-compose exec api python run_pipeline.py
    """)


def example_7_migration_from_old():
    """Пример 7: Миграция со старой системы."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 7: Миграция со старой системы")
    print("=" * 80)
    
    print("""
СТАРАЯ СИСТЕМА:
  LLM_PROFILE=free_openrouter python run_pipeline.py
  LLM_PROFILE=groq_free python run_pipeline.py
  LLM_PROFILE=auto_aggressive python run_pipeline.py

НОВАЯ СИСТЕМА (эквиваленты):
  LLM_PROVIDER=openrouter LLM_STRATEGY=cost_optimized python run_pipeline.py
  LLM_PROVIDER=groq python run_pipeline.py
  LLM_PROVIDER=groq LLM_FALLBACK_CHAIN=speed_first python run_pipeline.py

Или используйте готовые профили:
  LLM_PROFILE=free python run_pipeline.py
  LLM_PROFILE=dev python run_pipeline.py
  LLM_PROFILE=prod python run_pipeline.py
    """)


def run_all_examples():
    """Запустить все примеры."""
    examples = [
        example_1_simple_usage,
        example_2_fallback_chain,
        example_3_cost_optimization,
        example_4_quality_focus,
        example_5_programmatic,
        example_6_docker_compose,
        example_7_migration_from_old
    ]
    
    for example in examples:
        try:
            example()
            input("\n[Нажмите Enter для продолжения...]")
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем")
            break
        except Exception as e:
            print(f"\n❌ Ошибка в примере: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Примеры использования Models Config v4.0")
    parser.add_argument(
        "--example",
        type=int,
        choices=range(1, 8),
        help="Номер примера (1-7)"
    )
    
    args = parser.parse_args()
    
    if args.example:
        # Запустить конкретный пример
        examples = [
            None,  # 0
            example_1_simple_usage,
            example_2_fallback_chain,
            example_3_cost_optimization,
            example_4_quality_focus,
            example_5_programmatic,
            example_6_docker_compose,
            example_7_migration_from_old
        ]
        examples[args.example]()
    else:
        # Запустить все
        run_all_examples()
