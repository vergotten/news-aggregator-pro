# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/quality_validator_agent.py
# =============================================================================
"""
Агент валидации качества v8.0
"""

import logging
import re
from typing import Optional
from dataclasses import dataclass, field

from src.application.ai_services.agents.base_agent import BaseAgent, TaskType
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
    """Агент валидации качества нормализации."""

    agent_name = "quality_validator"
    task_type = TaskType.LIGHT

    # Критические паттерны
    CRITICAL_PATTERNS = [
        r'(?i)\[.*?(пропущено|truncated|cut off|incomplete).*?\]',
        r'(?i)^.{0,50}$',
    ]

    # Артефакты LLM
    LLM_ARTIFACTS = [
        r'^Вот переписанный текст:',
        r'^Результат:',
        r'^Here is the',
        r'^```',
    ]

    def __init__(self, llm_provider=None, config=None, **kwargs):
        super().__init__(llm_provider=llm_provider, config=config)
        logger.info(f"[INIT] QualityValidatorAgent v8")

    def validate(self, original: str, normalized: str, check_ai: bool = False) -> ValidationResult:
        """Валидировать качество."""
        return self.process(original, normalized, check_ai)

    def process(self, original: str, normalized: str, check_ai: bool = False) -> ValidationResult:
        """Главный метод валидации."""
        issues = []
        warnings = []
        metrics = {}

        orig_len = len(original)
        norm_len = len(normalized)
        
        metrics['original_length'] = orig_len
        metrics['normalized_length'] = norm_len

        # Пустой результат
        if norm_len < 50:
            issues.append(f"Результат слишком короткий: {norm_len} символов")
            return ValidationResult(
                is_valid=False, 
                score=0.0, 
                issues=issues, 
                warnings=warnings, 
                metrics=metrics
            )

        # Соотношение длин
        ratio = norm_len / orig_len if orig_len > 0 else 0
        metrics['length_ratio'] = round(ratio, 2)

        # Критическое сокращение (< 30%)
        if ratio < 0.3:
            issues.append(f"Критическое сокращение: {orig_len} -> {norm_len} (ratio: {ratio:.2f})")
        elif ratio < 0.5:
            warnings.append(f"Значительное сокращение: ratio={ratio:.2f}")

        # Критическое расширение (> 3x)
        if ratio > 3.0:
            issues.append(f"Критическое расширение: {orig_len} -> {norm_len} (ratio: {ratio:.2f})")
        elif ratio > 2.5:
            warnings.append(f"Значительное расширение: ratio={ratio:.2f}")

        # Маркеры обрезки
        for pattern in self.CRITICAL_PATTERNS:
            if re.search(pattern, normalized):
                issues.append("Обнаружены маркеры неполного текста")
                break

        # Артефакты LLM
        for pattern in self.LLM_ARTIFACTS:
            if re.search(pattern, normalized, re.IGNORECASE):
                warnings.append("Обнаружены артефакты LLM")
                break

        # Оценка
        score = 1.0
        score -= len(issues) * 0.4
        score -= len(warnings) * 0.1

        if 0.7 <= ratio <= 1.5:
            score += 0.1

        score = max(0.0, min(1.0, score))
        metrics['score'] = round(score, 2)

        is_valid = len(issues) == 0

        logger.info(
            f"[Validator] valid={is_valid}, score={score:.2f}, "
            f"issues={len(issues)}, warnings={len(warnings)}"
        )

        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            warnings=warnings,
            metrics=metrics
        )

    def quick_check(self, original: str, normalized: str) -> bool:
        """Быстрая проверка."""
        if len(normalized) < 50:
            return False
        
        if len(original) > 0 and len(normalized) / len(original) < 0.3:
            return False
        
        for pattern in self.CRITICAL_PATTERNS:
            if re.search(pattern, normalized):
                return False
        
        return True
