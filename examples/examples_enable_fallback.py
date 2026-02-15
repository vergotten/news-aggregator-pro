#!/usr/bin/env python3
"""
Примеры использования ENABLE_FALLBACK

Демонстрирует:
- Использование ТОЛЬКО одного провайдера
- Включение/выключение fallback
- Тестирование на конкретном провайдере
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models_config import get_models_config, reset_models_config


def example_single_provider():
    """Пример: ТОЛЬКО один провайдер (без fallback)."""
    print("\n" + "=" * 80)
    print("ПРИМЕР: ТОЛЬКО OpenRouter (без fallback)")
    print("=" * 80)
    
    reset_models_config()
    
    # Установить ТОЛЬКО OpenRouter
    os.environ["LLM_PROVIDER"] = "openrouter"
    os.environ["ENABLE_FALLBACK"] = "false"  # ВАЖНО!
    
    config = get_models_config()
    config.print_config()
    
    # Проверить fallback chain
    print("\nFallback chain:", config.get_fallback_providers())
    # Должен быть только: ['openrouter']
    
    assert len(config.get_fallback_providers()) == 1
    assert config.get_fallback_providers()[0] == "openrouter"
    
    print("\n✅ Успех! Используется ТОЛЬКО OpenRouter")


def example_with_fallback():
    """Пример: Провайдер + автоматический fallback."""
    print("\n" + "=" * 80)
    print("ПРИМЕР: Groq + автоматический fallback")
    print("=" * 80)
    
    reset_models_config()
    
    # Groq с fallback
    os.environ["LLM_PROVIDER"] = "groq"
    os.environ["ENABLE_FALLBACK"] = "true"  # Включить fallback
    
    config = get_models_config()
    config.print_config()
    
    # Проверить fallback chain
    print("\nFallback chain:", config.get_fallback_providers())
    # Должно быть несколько: ['groq', 'openrouter', 'google', 'ollama']
    
    assert len(config.get_fallback_providers()) > 1
    assert config.get_fallback_providers()[0] == "groq"
    
    print("\n✅ Успех! Groq с fallback")


def example_testing_specific_provider():
    """Пример: Тестирование конкретного провайдера."""
    print("\n" + "=" * 80)
    print("ПРИМЕР: Тестирование ТОЛЬКО Google (без fallback)")
    print("=" * 80)
    
    reset_models_config()
    
    # ТОЛЬКО Google для тестов
    os.environ["LLM_PROVIDER"] = "google"
    os.environ["ENABLE_FALLBACK"] = "false"
    os.environ["LLM_STRATEGY"] = "cost_optimized"  # Только бесплатные
    
    config = get_models_config()
    config.print_config()
    
    # Проверить что используются модели Google
    for agent in ["classifier", "summarizer", "rewriter"]:
        llm_config = config.get_llm_config(agent)
        print(f"  {agent:20} → {llm_config.model}")
        assert "gemini" in llm_config.model.lower()
    
    print("\n✅ Успех! Все агенты используют Google Gemini")


def example_docker_compose():
    """Пример: Конфигурация для Docker Compose."""
    print("\n" + "=" * 80)
    print("ПРИМЕР: Docker Compose конфигурация")
    print("=" * 80)
    
    print("""
# docker-compose.yml

services:
  api:
    environment:
      # Вариант 1: ТОЛЬКО Groq (для быстрых тестов)
      LLM_PROVIDER: groq
      ENABLE_FALLBACK: "false"
      GROQ_API_KEY: ${GROQ_API_KEY}
      
      # Вариант 2: Groq + fallback (для production)
      # LLM_PROVIDER: groq
      # ENABLE_FALLBACK: "true"
      # GROQ_API_KEY: ${GROQ_API_KEY}
      # OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}
      # GOOGLE_API_KEY: ${GOOGLE_API_KEY}
      
      # Вариант 3: ТОЛЬКО OpenRouter (тестирование платных моделей)
      # LLM_PROVIDER: openrouter
      # ENABLE_FALLBACK: "false"
      # LLM_STRATEGY: quality_focused
      # OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}

Запуск:

  # Только Groq
  ENABLE_FALLBACK=false LLM_PROVIDER=groq docker-compose exec api python run_pipeline.py
  
  # Groq + fallback
  ENABLE_FALLBACK=true LLM_PROVIDER=groq docker-compose exec api python run_pipeline.py
  
  # Только OpenRouter
  ENABLE_FALLBACK=false LLM_PROVIDER=openrouter docker-compose exec api python run_pipeline.py
    """)


def example_programmatic():
    """Пример: Программное управление."""
    print("\n" + "=" * 80)
    print("ПРИМЕР: Программное управление fallback")
    print("=" * 80)
    
    reset_models_config()
    
    # Вариант 1: ТОЛЬКО один провайдер
    print("\n--- Вариант 1: ТОЛЬКО Groq ---")
    config1 = get_models_config(
        provider="groq",
        enable_fallback=False  # Отключить fallback
    )
    print(f"Providers: {config1.get_fallback_providers()}")
    assert len(config1.get_fallback_providers()) == 1
    
    reset_models_config()
    
    # Вариант 2: С fallback
    print("\n--- Вариант 2: Groq + fallback ---")
    config2 = get_models_config(
        provider="groq",
        enable_fallback=True  # Включить fallback
    )
    print(f"Providers: {config2.get_fallback_providers()}")
    assert len(config2.get_fallback_providers()) > 1
    
    print("\n✅ Успех! Программное управление работает")


def example_smart_provider():
    """Пример: SmartLLMProvider с/без fallback."""
    print("\n" + "=" * 80)
    print("ПРИМЕР: SmartLLMProvider с/без fallback")
    print("=" * 80)
    
    from src.infrastructure.ai.smart_llm_provider import create_smart_provider
    
    reset_models_config()
    
    # Вариант 1: Без fallback
    print("\n--- Без fallback ---")
    os.environ["LLM_PROVIDER"] = "groq"
    os.environ["ENABLE_FALLBACK"] = "false"
    
    provider1 = create_smart_provider("classifier")
    print(f"Providers: {provider1.provider_names}")
    assert len(provider1.providers) == 1
    
    reset_models_config()
    
    # Вариант 2: С fallback
    print("\n--- С fallback ---")
    os.environ["ENABLE_FALLBACK"] = "true"
    
    provider2 = create_smart_provider("classifier")
    print(f"Providers: {provider2.provider_names}")
    assert len(provider2.providers) > 1
    
    print("\n✅ Успех! SmartLLMProvider корректно использует настройки")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ENABLE_FALLBACK EXAMPLES")
    print("=" * 80)
    
    try:
        example_single_provider()
        input("\n[Enter для продолжения...]")
        
        example_with_fallback()
        input("\n[Enter для продолжения...]")
        
        example_testing_specific_provider()
        input("\n[Enter для продолжения...]")
        
        example_docker_compose()
        input("\n[Enter для продолжения...]")
        
        example_programmatic()
        input("\n[Enter для продолжения...]")
        
        example_smart_provider()
        
        print("\n" + "=" * 80)
        print("✅ ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n👋 Прервано")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
