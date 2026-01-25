# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/quality_validator_agent.py
# =============================================================================
"""
Агент валидации качества обработки текста.

Версия 2.0 - Смягчённые правила для технических статей:
- Местоимения допустимы (авторский стиль)
- Допустимое изменение длины: 0.3x - 2.5x
- Фокус на сохранении смысла, а не формы
"""

import logging
import re
from typing import Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from src.application.ai_services.agents.base_agent import BaseAgent
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Результат валидации."""
    is_valid: bool
    score: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class QualityValidatorAgent(BaseAgent):
    """
    Агент валидации качества нормализации.

    Смягчённые правила v2.0:
    - Местоимения НЕ являются ошибкой (авторский стиль допустим)
    - Длина может меняться от 30% до 250% от оригинала
    - Основной критерий: текст не пустой и не обрезан критически
    """

    agent_name = "quality_validator"

    # Критические паттерны (реальные проблемы)
    CRITICAL_PATTERNS = [
        r'(?i)\[.*?(пропущено|truncated|cut off|incomplete).*?\]',  # Явные маркеры обрезки
        r'(?i)^.{0,50}$',  # Слишком короткий результат (< 50 символов)
    ]

    def __init__(self, llm_provider=None, config=None, ollama_client=None, **kwargs):
        if ollama_client:
            logger.warning("ollama_client deprecated")
        super().__init__(llm_provider=llm_provider, config=config)

    def validate(self, original: str, normalized: str, check_ai: bool = False) -> ValidationResult:
        """Валидировать качество."""
        return self.process(original, normalized, check_ai)

    def process(self, original: str, normalized: str, check_ai: bool = False) -> ValidationResult:
        """
        Главный метод валидации.

        Смягчённая логика:
        - issues = критические проблемы (текст обрезан, пустой)
        - warnings = некритичные замечания (информативно)
        - is_valid = True если нет критических проблем
        """
        issues, warnings, metrics = [], [], {}

        # Базовые проверки
        orig_len = len(original)
        norm_len = len(normalized)

        # Пустой или почти пустой результат - критическая ошибка
        if norm_len < 50:
            issues.append(f"Результат слишком короткий: {norm_len} символов")
            return ValidationResult(is_valid=False, score=0.0, issues=issues, warnings=warnings, metrics=metrics)

        # Соотношение длин
        ratio = norm_len / orig_len if orig_len > 0 else 0
        metrics['length_ratio'] = round(ratio, 2)
        metrics['original_length'] = orig_len
        metrics['normalized_length'] = norm_len

        # Критическое сокращение (меньше 30% от оригинала) - issue
        if ratio < 0.3:
            issues.append(f"Текст критически сокращён: {orig_len} → {norm_len} (ratio: {ratio:.2f})")
        # Сильное сокращение (30-50%) - только warning
        elif ratio < 0.5:
            warnings.append(f"Текст значительно сокращён: ratio={ratio:.2f}")

        # Критическое расширение (больше 3x) - issue
        if ratio > 3.0:
            issues.append(f"Текст критически расширен: {orig_len} → {norm_len} (ratio: {ratio:.2f})")
        # Сильное расширение (2.5-3x) - только warning
        elif ratio > 2.5:
            warnings.append(f"Текст значительно расширен: ratio={ratio:.2f}")

        # Проверка на маркеры обрезки
        for pattern in self.CRITICAL_PATTERNS:
            if re.search(pattern, normalized):
                issues.append("Обнаружены маркеры неполного текста")
                break

        # Расчёт оценки
        # Начинаем с 1.0, вычитаем за проблемы
        score = 1.0
        score -= len(issues) * 0.4  # Критические проблемы сильно снижают
        score -= len(warnings) * 0.1  # Warnings слабо влияют

        # Бонус за хорошее соотношение длин (0.7 - 1.5 идеально)
        if 0.7 <= ratio <= 1.5:
            score += 0.1

        score = max(0.0, min(1.0, score))
        metrics['score'] = round(score, 2)

        # Валидно если нет критических issues
        is_valid = len(issues) == 0

        result = ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            warnings=warnings,
            metrics=metrics
        )

        logger.info(
            f"Результат валидации: valid={is_valid}, score={score:.2f}, issues={len(issues)}, warnings={len(warnings)}")

        return result