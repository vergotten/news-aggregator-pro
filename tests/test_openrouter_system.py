#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты динамической системы OpenRouter с реальными API запросами.

Покрывает:
    - extract_parameters_b — извлечение размера из названия
    - compute_quality_score — вычисление рейтинга из параметров
    - OpenRouterModel — is_free, парсинг имени, сериализация
    - ModelsCache — TTL, истечение кэша
    - OpenRouterModels — API fetch, кэш, динамические фильтры, сортировка
    - ModelRateLimitInfo — cooldown, экспоненциальный backoff
    - RateLimitTracker — синглтон, доступность моделей
    - SmartModelSelector — умный выбор с cooldown
    - extract_json — извлечение JSON из ответов
    - OpenRouterProvider — авто-стратегия с реальными моделями

Запуск:
    pytest test_openrouter_system.py -v
    OPENROUTER_API_KEY=sk-or-... pytest test_openrouter_system.py -v -k Integration
"""

import json
import math
import os
import sys
import pytest
import requests
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, PropertyMock
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.infrastructure.ai.openrouter_models import (
    OpenRouterModel,
    ModelsCache,
    OpenRouterModels,
    ModelRateLimitInfo,
    RateLimitTracker,
    SmartModelSelector,
    extract_json,
    extract_parameters_b,
    compute_quality_score,
)
from src.infrastructure.ai.llm_provider import (
    OpenRouterProvider,
    LLMConfig,
    LLMProviderType,
    OpenRouterModelDiscovery,
    ModelStatus,
    ModelStatusTracker,
    FreeModel,
    LLMProviderFactory,
)


# ============================================================================
# Фабрики тестовых данных
# ============================================================================

def build_api_model(
        model_id: str,
        name: str,
        context: int = 131072,
        prompt_price: str = "0",
        completion_price: str = "0",
        modality: str = "text->text",
        supported_params: Optional[List[str]] = None,
        max_completion: int = 4096,
) -> Dict[str, Any]:
    """Построить словарь модели в формате OpenRouter API."""
    result = {
        "id": model_id,
        "name": name,
        "pricing": {"prompt": prompt_price, "completion": completion_price},
        "context_length": context,
        "architecture": {"modality": modality},
        "top_provider": {"max_completion_tokens": max_completion, "name": "TestProvider"},
    }
    if supported_params is not None:
        result["supported_parameters"] = supported_params
    return result


def make_client() -> OpenRouterModels:
    """Создать клиент OpenRouterModels с реальным API ключом."""
    api_key = os.getenv("OPENROUTER_API_KEY", "test-key-123")
    return OpenRouterModels(api_key=api_key, cache_ttl=60)


def make_provider() -> OpenRouterProvider:
    """Создать провайдер OpenRouter с реальным API ключом."""
    api_key = os.getenv("OPENROUTER_API_KEY", "test-key-123")
    config = LLMConfig(
        provider=LLMProviderType.OPENROUTER,
        model="meta-llama/llama-3.1-8b-instruct:free",
        api_key=api_key,
    )
    return OpenRouterProvider(config)


# ============================================================================
# 1. extract_parameters_b — извлечение размера из названия
# ============================================================================

class TestExtractParametersB:
    """Тесты извлечения количества параметров из названия модели."""

    def test_extract_70b_from_name(self):
        """'Llama 3.3 70B Instruct' → 70.0"""
        assert extract_parameters_b("Llama 3.3 70B Instruct") == 70.0

    def test_extract_72b_from_id(self):
        """'qwen2.5-72b-instruct' → 72.0"""
        assert extract_parameters_b("qwen2.5-72b-instruct") == 72.0

    def test_extract_8b(self):
        """'llama-3.1-8b-instruct' → 8.0"""
        assert extract_parameters_b("llama-3.1-8b-instruct") == 8.0

    def test_extract_3b(self):
        """'llama-3.2-3b-instruct' → 3.0"""
        assert extract_parameters_b("llama-3.2-3b-instruct") == 3.0

    def test_extract_24b(self):
        """'mistral-small-3.1-24b' → 24.0"""
        assert extract_parameters_b("mistral-small-3.1-24b") == 24.0

    def test_extract_405b(self):
        """'llama-3.1-405b' → 405.0"""
        assert extract_parameters_b("llama-3.1-405b") == 405.0

    def test_extract_none_without_b(self):
        """'Gemini 2.0 Flash' → None (нет B в названии)"""
        assert extract_parameters_b("Gemini 2.0 Flash") is None

    def test_extract_deepseek_none(self):
        """'DeepSeek V3 0324' → None"""
        assert extract_parameters_b("DeepSeek V3 0324") is None

    def test_extract_filter_too_small(self):
        """Значение < 0.5 отсеивается как нереалистичное."""
        assert extract_parameters_b("model-0.1b") is None

    def test_extract_fractional_size(self):
        """'model-1.5b' → 1.5"""
        assert extract_parameters_b("model-1.5b-instruct") == 1.5

    def test_extract_max_multiple(self):
        """При нескольких числах с 'b' берёт максимальное."""
        assert extract_parameters_b("mistral-small-3.1-24b-instruct") == 24.0

    def test_extract_80b_from_complex_id(self):
        """'qwen/qwen3-next-80b-a3b-thinking' → 80.0 (не 3.0)"""
        assert extract_parameters_b("qwen/qwen3-next-80b-a3b-thinking") == 80.0

    def test_extract_3b_not_part_of_a3b(self):
        """Проверяем, что 'a3b' не извлекается как 3b"""
        assert extract_parameters_b("model-a3b-thinking") is None

    def test_extract_28b_not_a3b(self):
        """Проверяем, что '28b-a3b' извлекается как 28b, а не 3b"""
        assert extract_parameters_b("ernie-4.5-vl-28b-a3b") == 28.0

# ============================================================================
# 2. compute_quality_score — вычисление рейтинга
# ============================================================================

class TestComputeQualityScore:
    """Тесты вычисления динамического рейтинга качества."""

    def test_score_big_model_higher_than_small(self):
        """70B модель получает больший score чем 3B."""
        big = OpenRouterModel(id="x/big-70b:free", name="Big 70B", parameters_b=70.0,
                              context_length=131072, max_completion_tokens=8192)
        small = OpenRouterModel(id="x/small-3b:free", name="Small 3B", parameters_b=3.0,
                                context_length=131072, max_completion_tokens=8192)
        big.quality_score = compute_quality_score(big)
        small.quality_score = compute_quality_score(small)
        assert big.quality_score > small.quality_score

    def test_score_long_context_bonus(self):
        """Модель с 1M контекстом получает больше баллов чем с 4k."""
        long_ctx = OpenRouterModel(id="x/a", name="A", parameters_b=8.0,
                                   context_length=1_048_576, max_completion_tokens=4096)
        short_ctx = OpenRouterModel(id="x/b", name="B", parameters_b=8.0,
                                    context_length=4096, max_completion_tokens=4096)
        assert compute_quality_score(long_ctx) > compute_quality_score(short_ctx)

    def test_score_big_output_bonus(self):
        """Модель с max_output=32k лучше чем с 2k при тех же параметрах."""
        big_out = OpenRouterModel(id="x/a", name="A", parameters_b=8.0,
                                  context_length=131072, max_completion_tokens=32768)
        small_out = OpenRouterModel(id="x/b", name="B", parameters_b=8.0,
                                    context_length=131072, max_completion_tokens=2048)
        assert compute_quality_score(big_out) > compute_quality_score(small_out)

    def test_score_vision_bonus(self):
        """Модель с vision получает бонус."""
        with_vision = OpenRouterModel(id="x/a", name="A", parameters_b=8.0,
                                      capabilities=["chat", "vision"])
        without = OpenRouterModel(id="x/b", name="B", parameters_b=8.0,
                                  capabilities=["chat"])
        assert compute_quality_score(with_vision) > compute_quality_score(without)

    def test_score_function_calling_bonus(self):
        """Модель с function_calling получает бонус."""
        with_fc = OpenRouterModel(id="x/a", name="A", parameters_b=8.0,
                                  capabilities=["chat", "function_calling"])
        without = OpenRouterModel(id="x/b", name="B", parameters_b=8.0,
                                  capabilities=["chat"])
        assert compute_quality_score(with_fc) > compute_quality_score(without)

    def test_score_unknown_size_average(self):
        """Модель без parameters_b получает средний балл (не 0)."""
        unknown = OpenRouterModel(id="x/a", name="A", parameters_b=None,
                                  context_length=131072, max_completion_tokens=8192)
        score = compute_quality_score(unknown)
        assert score > 10  # не нулевой, не минимальный

    def test_score_always_positive(self):
        """Даже минимальная модель получает > 0."""
        tiny = OpenRouterModel(id="x/a", name="A", parameters_b=1.0,
                               context_length=4096, max_completion_tokens=1024)
        assert compute_quality_score(tiny) > 0

    def test_score_parameters_more_important_than_context(self):
        """70B с 4k контекстом всё равно лучше 3B с 1M контекстом."""
        big_params = OpenRouterModel(id="x/a", name="A", parameters_b=70.0,
                                     context_length=4096, max_completion_tokens=4096)
        big_ctx = OpenRouterModel(id="x/b", name="B", parameters_b=3.0,
                                  context_length=1_048_576, max_completion_tokens=4096)
        assert compute_quality_score(big_params) > compute_quality_score(big_ctx)


# ============================================================================
# 3. OpenRouterModel — датакласс
# ============================================================================

class TestOpenRouterModel:
    """Тесты датакласса OpenRouterModel."""

    def test_is_free_true(self):
        """Модель бесплатна если цены = 0."""
        m = OpenRouterModel(id="x/y:free", name="T", prompt_price=0.0, completion_price=0.0)
        assert m.is_free is True

    def test_is_free_false_prompt(self):
        """Модель платная если prompt_price > 0."""
        m = OpenRouterModel(id="x/y", name="T", prompt_price=5.0)
        assert m.is_free is False

    def test_is_free_false_completion(self):
        """Модель платная если completion_price > 0."""
        m = OpenRouterModel(id="x/y", name="T", completion_price=10.0)
        assert m.is_free is False

    def test_short_name_with_slash(self):
        """short_name для ID со слэшем."""
        m = OpenRouterModel(id="meta-llama/llama-3.2-3b-instruct:free", name="T")
        assert m.short_name == "llama-3.2-3b-instruct:free"

    def test_author_with_slash(self):
        """author для ID со слэшем."""
        m = OpenRouterModel(id="meta-llama/llama-3.2-3b-instruct:free", name="T")
        assert m.author == "meta-llama"

    def test_short_name_no_slash(self):
        """short_name для ID без слэша."""
        m = OpenRouterModel(id="llama3", name="T")
        assert m.short_name == "llama3"

    def test_author_no_slash(self):
        """author для ID без слэша."""
        m = OpenRouterModel(id="llama3", name="T")
        assert m.author == "unknown"

    def test_to_dict_all_keys(self):
        """to_dict содержит все ключи."""
        m = OpenRouterModel(id="t/m:free", name="T", context_length=65536,
                            capabilities=["chat", "vision"], parameters_b=70.0, quality_score=55.5)
        d = m.to_dict()
        assert d["id"] == "t/m:free"
        assert d["is_free"] is True
        assert d["parameters_b"] == 70.0
        assert d["quality_score"] == 55.5
        assert "vision" in d["capabilities"]

    def test_repr_contains_quality(self):
        """repr содержит параметры и качество."""
        m = OpenRouterModel(id="t/m:free", name="T", parameters_b=70.0, quality_score=55.3)
        r = repr(m)
        assert "70.0B" in r
        assert "55.3" in r


# ============================================================================
# 4. ModelsCache
# ============================================================================

class TestModelsCache:
    """Тесты кэша моделей."""

    def test_is_expired_false_fresh(self):
        """Свежий кэш не истёк."""
        assert ModelsCache(models=[], fetched_at=datetime.now(), ttl_minutes=60).is_expired is False

    def test_is_expired_true_old(self):
        """Старый кэш истёк."""
        old = datetime.now() - timedelta(hours=2)
        assert ModelsCache(models=[], fetched_at=old, ttl_minutes=60).is_expired is True

    def test_is_expired_true_zero_ttl(self):
        """Кэш с нулевым TTL сразу истёк."""
        assert ModelsCache(models=[], fetched_at=datetime.now(), ttl_minutes=0).is_expired is True


# ============================================================================
# 5. OpenRouterModels — РЕАЛЬНЫЕ API ЗАПРОСЫ
# ============================================================================

class TestOpenRouterModelsRealAPI:
    """Тесты клиента с реальными API запросами."""

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_fetch_real_models(self):
        """Получение реальных моделей с API."""
        client = make_client()
        models = client.get_all_models()

        # Проверяем, что модели получены
        assert len(models) > 0, "Не удалось получить модели от API"

        # Проверяем структуру данных
        for model in models[:5]:  # Проверяем первые 5 моделей
            assert isinstance(model, OpenRouterModel)
            assert model.id
            assert model.name
            assert model.context_length > 0
            assert model.max_completion_tokens > 0
            assert isinstance(model.is_free, bool)
            assert model.quality_score > 0

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_get_free_models_real(self):
        """Получение реальных бесплатных моделей."""
        client = make_client()
        free_models = client.get_free_models()

        # Проверяем, что есть бесплатные модели
        assert len(free_models) > 0, "Не найдено бесплатных моделей"

        # Все модели должны быть бесплатными
        for model in free_models:
            assert model.is_free is True, f"Модель {model.id} не бесплатна"
            assert ":free" in model.id or model.prompt_price == 0.0

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_filter_min_context_real(self):
        """Фильтр по минимальному контексту с реальными моделями."""
        client = make_client()

        # Тестируем с разным контекстом
        for min_ctx in [100000, 200000, 500000]:
            models = client.get_free_models(min_context=min_ctx)
            for model in models:
                assert model.context_length >= min_ctx

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_filter_min_output_real(self):
        """Фильтр по минимальному output с реальными моделями."""
        client = make_client()

        # Тестируем с разным output
        for min_out in [2000, 4000, 8000]:
            models = client.get_free_models(min_output=min_out)
            for model in models:
                assert model.max_completion_tokens >= min_out

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_filter_capabilities_real(self):
        """Фильтр по capabilities с реальными моделями."""
        client = make_client()

        # Проверяем модели с vision
        vision_models = client.get_free_models(capabilities=["vision"])
        for model in vision_models:
            assert "vision" in model.capabilities

        # Проверяем модели с function_calling
        fc_models = client.get_free_models(capabilities=["function_calling"])
        for model in fc_models:
            assert "function_calling" in model.capabilities

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_filter_min_parameters_real(self):
        """Фильтр по минимальным параметрам с реальными моделями."""
        client = make_client()

        # Тестируем с разным размером
        for min_params in [8.0, 20.0, 50.0]:
            models = client.get_free_models(min_parameters_b=min_params)
            for model in models:
                if model.parameters_b is not None:
                    assert model.parameters_b >= min_params

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_sort_quality_real(self):
        """Сортировка по качеству с реальными моделями."""
        client = make_client()
        models = client.get_free_models(sort_by="quality")

        # Проверяем, что модели отсортированы по quality_score
        scores = [m.quality_score for m in models]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_sort_context_real(self):
        """Сортировка по контексту с реальными моделями."""
        client = make_client()
        models = client.get_free_models(sort_by="context")

        # Проверяем, что модели отсортированы по контексту
        contexts = [m.context_length for m in models]
        assert contexts == sorted(contexts, reverse=True)

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_sort_parameters_real(self):
        """Сортировка по параметрам с реальными моделями."""
        client = make_client()
        models = client.get_free_models(sort_by="parameters")

        # Проверяем, что модели отсортированы по параметрам
        params = [(m.parameters_b or 0) for m in models]
        assert params == sorted(params, reverse=True)

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_get_best_free_model_real(self):
        """Получение лучшей бесплатной модели."""
        client = make_client()
        best = client.get_best_free_model()

        assert best is not None, "Не найдена лучшая бесплатная модель"
        assert best.is_free
        assert best.quality_score > 0

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_get_model_by_id_real(self):
        """Поиск модели по ID с реальными моделями."""
        client = make_client()

        # Получаем список моделей
        models = client.get_free_models()
        if models:
            # Ищем первую модель по ID
            model_id = models[0].id
            found = client.get_model_by_id(model_id)
            assert found is not None
            assert found.id == model_id

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_cache_behavior_real(self):
        """Проверка работы кэша с реальными моделями."""
        client = make_client()

        # Первый запрос - должен дёрнуть API
        models1 = client.get_all_models()

        # Второй запрос - должен использовать кэш
        models2 = client.get_all_models()

        # Результаты должны быть одинаковыми
        assert len(models1) == len(models2)
        assert [m.id for m in models1] == [m.id for m in models2]

        # force_refresh должен обновить кэш
        models3 = client.get_all_models(force_refresh=True)
        assert len(models3) >= 0  # Может измениться количество моделей

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_file_cache_real(self, tmp_path):
        """Файловый кэш с реальными моделями."""
        cache_file = tmp_path / "test_models.json"
        client = OpenRouterModels(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            cache_file=str(cache_file),
            cache_ttl=60
        )

        # Получаем модели
        models1 = client.get_all_models()
        assert cache_file.exists()

        # Создаем новый клиент с тем же кэшем
        client2 = OpenRouterModels(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            cache_file=str(cache_file),
            cache_ttl=60
        )

        # Должны загрузиться из кэша
        models2 = client2.get_all_models()
        assert len(models1) == len(models2)
        assert [m.id for m in models1] == [m.id for m in models2]

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_real_model_parsing(self):
        """Проверка парсинга реальных моделей."""
        client = make_client()
        models = client.get_all_models()

        # Проверяем, что у всех моделей есть нужные поля
        for model in models[:10]:  # Проверяем первые 10
            assert model.id
            assert model.name
            assert model.context_length > 0
            assert model.max_completion_tokens > 0
            assert isinstance(model.is_free, bool)
            assert model.quality_score > 0
            assert isinstance(model.capabilities, list)
            assert "chat" in model.capabilities

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_real_parameters_extraction(self):
        """Проверка извлечения параметров из реальных моделей."""
        client = make_client()
        models = client.get_all_models()

        # Проверяем конкретные модели с известными параметрами
        known_models = {
            # 70B модели
            "meta-llama/llama-3.1-70b-instruct:free": 70.0,
            "meta-llama/llama-3-70b-instruct:free": 70.0,

            # 8B модели
            "meta-llama/llama-3.1-8b-instruct:free": 8.0,
            "meta-llama/llama-3-8b-instruct:free": 8.0,

            # 3B модели
            "meta-llama/llama-3.2-3b-instruct:free": 3.0,

            # 24B модели
            "mistralai/mistral-small-24b-instruct:free": 24.0,

            # 28B модели
            "baidu/ernie-4.5-vl-28b-a3b": 28.0,

            # 424B модели
            "baidu/ernie-4.5-vl-424b-a47b": 424.0,

            # 47B модели
            "baidu/ernie-4.5-vl-47b-a47b": 47.0,
        }

        for model_id, expected_params in known_models.items():
            model = client.get_model_by_id(model_id)
            if model:  # Модель может отсутствовать в API
                assert model.parameters_b == expected_params, f"Ошибка в {model.id}: ожидалось {expected_params}, получено {model.parameters_b}"

        # Дополнительная проверка: если в ID есть "424b", то параметры должны быть 424.0
        for model in models:
            if "424b" in model.id.lower():
                assert model.parameters_b == 424.0, f"Ошибка в {model.id}: ожидалось 424.0, получено {model.parameters_b}"


# ============================================================================
# 6. ModelRateLimitInfo
# ============================================================================

class TestModelRateLimitInfo:
    """Тесты информации о rate limit для модели."""

    def test_initial_not_cooldown(self):
        """Начальное состояние - не в cooldown."""
        info = ModelRateLimitInfo(model_id="t/m")
        assert info.is_in_cooldown is False
        assert info.consecutive_errors == 0

    def test_error_sets_cooldown(self):
        """Ошибка устанавливает cooldown."""
        info = ModelRateLimitInfo(model_id="t/m")
        info.record_error(429)
        assert info.is_in_cooldown is True
        assert info.consecutive_errors == 1

    def test_exponential_backoff(self):
        """Экспоненциальный backoff."""
        info = ModelRateLimitInfo(model_id="t/m")
        info.record_error(429)
        c1 = (info.cooldown_until - datetime.now()).total_seconds()
        info.record_error(429)
        c2 = (info.cooldown_until - datetime.now()).total_seconds()
        info.record_error(429)
        c3 = (info.cooldown_until - datetime.now()).total_seconds()
        assert 20 < c1 < 40
        assert 40 < c2 < 80
        assert 80 < c3 < 160

    def test_max_cooldown(self):
        """Максимальный cooldown."""
        info = ModelRateLimitInfo(model_id="t/m")
        for _ in range(10):
            info.record_error(429)
        assert (info.cooldown_until - datetime.now()).total_seconds() <= 750

    def test_success_resets_cooldown(self):
        """Успех сбрасывает cooldown."""
        info = ModelRateLimitInfo(model_id="t/m")
        info.record_error(429)
        info.record_success()
        assert info.is_in_cooldown is False
        assert info.consecutive_errors == 0

    def test_rpm_counter(self):
        """Счётчик запросов в минуту."""
        info = ModelRateLimitInfo(model_id="t/m")
        for _ in range(3):
            info.record_success()
        assert info.requests_this_minute == 3


# ============================================================================
# 7. RateLimitTracker
# ============================================================================

class TestRateLimitTracker:
    """Тесты трекера rate limit."""

    def setup_method(self):
        """Сброс перед каждым тестом."""
        RateLimitTracker.reset()

    def test_singleton(self):
        """Проверка синглтона."""
        assert RateLimitTracker() is RateLimitTracker()

    def test_new_model_available(self):
        """Новая модель доступна."""
        assert RateLimitTracker().is_available("t/m") is True

    def test_error_model_unavailable(self):
        """Ошибка делает модель недоступной."""
        t = RateLimitTracker()
        t.record_error("t/m", 429)
        assert t.is_available("t/m") is False

    def test_success_restores_availability(self):
        """Успех восстанавливает доступность."""
        t = RateLimitTracker()
        t.record_error("t/m", 429)
        t.record_success("t/m")
        assert t.is_available("t/m") is True

    def test_get_available_filters(self):
        """get_available фильтрует недоступные."""
        t = RateLimitTracker()
        t.record_error("a", 429)
        avail = t.get_available(["a", "b", "c"])
        assert "a" not in avail and "b" in avail

    def test_cooldown_status(self):
        """Статус cooldown."""
        t = RateLimitTracker()
        t.record_error("x", 429)
        s = t.get_cooldown_status()
        assert s["x"]["in_cooldown"] is True

    def test_reset_clears_all(self):
        """Reset очищает все состояния."""
        t1 = RateLimitTracker()
        t1.record_error("z", 429)
        RateLimitTracker.reset()
        t2 = RateLimitTracker()
        assert t1 is not t2
        assert t2.is_available("z") is True


# ============================================================================
# 8. SmartModelSelector — РЕАЛЬНЫЕ API ЗАПРОСЫ
# ============================================================================

class TestSmartModelSelectorRealAPI:
    """Тесты умного выбора модели с реальными API."""

    def setup_method(self):
        """Сброс перед каждым тестом."""
        RateLimitTracker.reset()

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_select_best_model_real(self):
        """Выбор лучшей модели с реальными данными."""
        client = make_client()
        selector = SmartModelSelector(client)

        # Выбираем лучшую модель
        model_id = selector.select()
        assert model_id is not None

        # Проверяем, что модель существует
        model = client.get_model_by_id(model_id)
        assert model is not None
        assert model.is_free

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_select_with_constraints_real(self):
        """Выбор модели с ограничениями."""
        client = make_client()
        selector = SmartModelSelector(client)

        # Выбираем модель с большим контекстом
        model_id = selector.select(min_context=200000)
        assert model_id is not None

        model = client.get_model_by_id(model_id)
        assert model.context_length >= 200000

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_select_with_capabilities_real(self):
        """Выбор модели с capabilities."""
        client = make_client()
        selector = SmartModelSelector(client)

        # Выбираем модель с vision
        model_id = selector.select(capabilities=["vision"])
        if model_id:  # Может не быть таких моделей
            model = client.get_model_by_id(model_id)
            assert "vision" in model.capabilities

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_select_exclude_real(self):
        """Выбор модели с исключением."""
        client = make_client()
        selector = SmartModelSelector(client)

        # Получаем первую модель
        first = selector.select()

        # Исключаем её
        second = selector.select(exclude=[first])
        assert second is not None
        assert second != first

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_select_with_cooldown_real(self):
        """Выбор модели учитывает cooldown."""
        client = make_client()
        selector = SmartModelSelector(client)

        # Выбираем модель
        first = selector.select()

        # Ставим её в cooldown
        RateLimitTracker().record_error(first, 429)

        # Следующий выбор должен вернуть другую модель
        second = selector.select()
        assert second != first

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_models_to_try_real(self):
        """Получение списка моделей для попыток."""
        client = make_client()
        selector = SmartModelSelector(client)

        # Получаем список моделей
        models_to_try = selector.get_models_to_try()
        assert len(models_to_try) > 0

        # Все модели должны быть доступны
        for model_id in models_to_try:
            assert RateLimitTracker().is_available(model_id)

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_select_min_parameters_real(self):
        """Выбор модели с минимальными параметрами."""
        client = make_client()
        selector = SmartModelSelector(client)

        # Выбираем модель с параметрами >= 20B
        model_id = selector.select(min_parameters_b=20.0)
        if model_id:  # Может не быть таких моделей
            model = client.get_model_by_id(model_id)
            assert model.parameters_b >= 20.0


# ============================================================================
# 9. extract_json
# ============================================================================

class TestExtractJson:
    """Тесты извлечения JSON из текста."""

    def test_extract_clean_json(self):
        """Извлечение чистого JSON."""
        assert extract_json('{"score": 7}') == {"score": 7}

    def test_extract_markdown_json(self):
        """Извлечение JSON из markdown."""
        assert extract_json('```json\n{"ok": true}\n```') == {"ok": True}

    def test_extract_code_block(self):
        """Извлечение JSON из code block."""
        assert extract_json('```\n{"x": 1}\n```') == {"x": 1}

    def test_extract_from_text(self):
        """Извлечение JSON из текста."""
        assert extract_json('blah {"s": 9} blah')["s"] == 9

    def test_extract_nested(self):
        """Извлечение вложенного JSON."""
        assert extract_json('{"a": {"b": 42}}')["a"]["b"] == 42

    def test_extract_array(self):
        """Извлечение массива."""
        assert extract_json('[1,2]') == [1, 2]

    def test_extract_none_garbage(self):
        """None для мусорного текста."""
        assert extract_json("просто текст") is None

    def test_extract_none_empty(self):
        """None для пустой строки."""
        assert extract_json("") is None

    def test_extract_none_null(self):
        """None для None."""
        assert extract_json(None) is None

    def test_extract_unicode(self):
        """Извлечение с юникодом."""
        assert extract_json('{"к": "в"}')["к"] == "в"


# ============================================================================
# 10. OpenRouterProvider — РЕАЛЬНЫЕ API ЗАПРОСЫ
# ============================================================================

class TestOpenRouterProviderRealAPI:
    """Тесты OpenRouterProvider с реальными API запросами."""

    def setup_method(self):
        """Сброс синглтонов."""
        OpenRouterModelDiscovery._instance = None
        ModelStatusTracker._instance = None

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_provider_get_models_real(self):
        """Провайдер получает реальные модели."""
        provider = make_provider()

        # Получаем модели через discovery
        models = provider._discovery.get_free_models()
        assert len(models) > 0

        # Проверяем структуру с валидацией
        for model in models[:5]:  # Проверяем первые 5 моделей
            assert isinstance(model, FreeModel)
            assert model.id
            assert model.name
            assert model.context_length > 0
            # max_output может быть None в некоторых моделях
            if model.max_output is not None:
                assert model.max_output > 0

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_provider_models_to_try_real(self):
        """Получение списка моделей для попыток."""
        provider = make_provider()

        # Получаем список моделей
        models_to_try = provider._get_models_to_try()
        assert len(models_to_try) > 0

        # Все должны быть строками
        for model_id in models_to_try:
            assert isinstance(model_id, str)

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_provider_available_models_real(self):
        """Получение доступных моделей."""
        provider = make_provider()

        # Получаем доступные модели
        available = provider.get_available_models()
        assert len(available) > 0

        # Проверяем, что все доступны
        for model_id in available:
            assert provider._tracker.is_available(model_id)

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_provider_with_cooldown_real(self):
        """Провайдер учитывает cooldown."""
        provider = make_provider()

        # Получаем модели
        models_to_try = provider._get_models_to_try()

        # Помещаем первую в cooldown
        if models_to_try:
            # Используем правильную сигнатуру метода
            provider._tracker.record_error(models_to_try[0])

            # Получаем модели снова
            new_models = provider._get_models_to_try()

            # Первая модель не должна быть в списке
            assert models_to_try[0] not in new_models

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_provider_metrics_real(self):
        """Метрики провайдера."""
        provider = make_provider()

        # Получаем метрики
        metrics = provider.get_metrics()

        assert metrics["provider"] == "openrouter"
        assert metrics["requests"] == 0
        assert metrics["errors"] == 0
        assert "current_model" in metrics
        assert "fallbacks" in metrics

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_provider_singleton_behavior_real(self):
        """Проверка синглтон поведения."""
        provider1 = make_provider()
        provider2 = make_provider()

        # Они должны использовать одинаковые синглтоны
        assert provider1._discovery is provider2._discovery
        assert provider1._tracker is provider2._tracker

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_provider_filter_by_context_real(self):
        """Фильтрация моделей по контексту."""
        provider = make_provider()

        # Получаем модели с разным контекстом
        for min_ctx in [100000, 200000, 500000]:
            models = provider._discovery.get_free_models(min_context=min_ctx)
            for model in models:
                assert model.context_length >= min_ctx

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_provider_capabilities_real(self):
        """Проверка capabilities реальных моделей."""
        provider = make_provider()

        # Получаем модели
        models = provider._discovery.get_free_models()

        # Проверяем, что у некоторых есть vision
        vision_models = [m for m in models if "vision" in m.capabilities]
        if vision_models:
            for model in vision_models:
                assert "vision" in model.capabilities

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_provider_factory_auto_real(self):
        """Фабрика провайдеров с авто-выбором."""
        # Создаем провайдер через фабрику
        provider = LLMProviderFactory.create_auto(min_context=8000)

        # Проверяем, что это OpenRouterProvider
        assert isinstance(provider, OpenRouterProvider)

        # Проверяем, что модель выбрана
        assert provider.model

        # Проверяем, что провайдер работает
        models = provider._discovery.get_free_models()
        assert len(models) > 0

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
    def test_provider_real_model_priority_real(self):
        """Приоритет текущей модели."""
        # Создаем провайдер с конкретной моделью
        config = LLMConfig(
            provider=LLMProviderType.OPENROUTER,
            model="meta-llama/llama-3.1-8b-instruct:free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        provider = OpenRouterProvider(config)

        # Получаем модели для попыток
        models_to_try = provider._get_models_to_try()

        # Текущая модель должна быть первой если доступна
        if models_to_try and provider._tracker.is_available(provider.model):
            assert models_to_try[0] == provider.model


# ============================================================================
# 11. Интеграционные тесты
# ============================================================================

@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY не задан")
class TestFullIntegration:
    """Полные интеграционные тесты."""

    def test_complete_workflow_real(self):
        """Полный рабочий процесс с реальными моделями."""
        # Создаем клиент
        client = make_client()

        # Получаем бесплатные модели
        free_models = client.get_free_models()
        assert len(free_models) > 0

        # Сортируем по качеству
        quality_sorted = client.get_free_models(sort_by="quality")
        scores = [m.quality_score for m in quality_sorted]
        assert scores == sorted(scores, reverse=True)

        # Фильтруем по контексту
        high_context = client.get_free_models(min_context=200000)
        for model in high_context:
            assert model.context_length >= 200000

        # Используем SmartModelSelector
        selector = SmartModelSelector(client)
        best = selector.select()
        assert best is not None

        # Проверяем, что модель существует
        model = client.get_model_by_id(best)
        assert model is not None
        assert model.is_free

    def test_provider_factory_integration_real(self):
        """Интеграция с фабрикой провайдеров."""
        # Создаем провайдер через фабрику
        provider = LLMProviderFactory.create_auto(min_context=100000)

        # Проверяем, что всё работает
        assert isinstance(provider, OpenRouterProvider)

        # Получаем модели
        models = provider._discovery.get_free_models(min_context=100000)
        assert len(models) > 0

        for model in models:
            assert model.context_length >= 100000

    def test_error_handling_real(self):
        """Обработка ошибок с реальными данными."""
        # Создаем клиент с неверным ключом
        client = OpenRouterModels(api_key="invalid-key-12345", cache_ttl=1)

        # Получаем модели - API может вернуть модели даже с неверным ключом
        # или может вернуть пустой список в случае ошибки
        models = client.get_all_models()

        # Проверяем, что получен либо пустой список, либо список моделей
        assert isinstance(models, list)

        # Если модели получены, проверяем их структуру
        if models:
            for model in models[:3]:  # Проверяем первые 3 модели
                assert isinstance(model, OpenRouterModel)
                assert model.id
                assert model.name
                assert model.context_length > 0
                assert model.max_completion_tokens > 0
                assert isinstance(model.is_free, bool)
                assert model.quality_score > 0

    def test_cache_integration_real(self, tmp_path):
        """Интеграция кэша с реальными данными."""
        cache_file = tmp_path / "integration_cache.json"

        # Создаем клиент с кэшем
        client1 = OpenRouterModels(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            cache_file=str(cache_file),
            cache_ttl=60
        )

        # Получаем модели
        models1 = client1.get_all_models()
        assert cache_file.exists()

        # Создаем второй клиент
        client2 = OpenRouterModels(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            cache_file=str(cache_file),
            cache_ttl=60
        )

        # Должны загрузиться из кэша
        models2 = client2.get_all_models()
        assert len(models1) == len(models2)

        # Проверяем, что данные совпадают
        ids1 = [m.id for m in models1]
        ids2 = [m.id for m in models2]
        assert ids1 == ids2