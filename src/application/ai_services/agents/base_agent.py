# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/base_agent.py
# =============================================================================
"""
Базовый класс для AI агентов v9.0

Изменения v9:
- Обработка ошибок context length
- Автоматическое определение лимитов модели
- Улучшенная обработка пустых ответов
"""

from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar, Any
import logging
import time
import re
from dataclasses import dataclass

from pydantic import BaseModel

from src.infrastructure.ai.llm_provider import (
    LLMProvider,
    LLMProviderFactory,
    LLMProviderType,
    TaskType
)
from src.config.models_config import ModelsConfig, get_models_config

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

__all__ = ['BaseAgent', 'TaskType', 'AgentMetrics', 'ContextLimitError']


class ContextLimitError(Exception):
    """Ошибка превышения лимита контекста."""
    def __init__(self, message: str, requested: int = 0, limit: int = 0):
        super().__init__(message)
        self.requested = requested
        self.limit = limit


@dataclass
class AgentMetrics:
    """Метрики агента."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    empty_responses: int = 0
    context_errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successful_calls / self.total_calls if self.total_calls else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.successful_calls if self.successful_calls else 0.0


class BaseAgent(ABC):
    """
    Базовый класс для AI агентов v9.0
    """

    agent_name: str = "base"
    task_type: TaskType = TaskType.MEDIUM
    MIN_RESPONSE_LENGTH: int = 10
    
    # Примерные лимиты контекста (символы) для разных моделей
    MODEL_CONTEXT_LIMITS = {
        'default': 30000,          # ~8K токенов
        'llama-70b': 50000,        # ~12K токенов
        'llama-405b': 100000,      # ~25K токенов
        'qwen-235b': 200000,       # ~50K токенов
        'gemini': 200000,
    }

    def __init__(
            self,
            llm_provider: Optional[LLMProvider] = None,
            config: Optional[ModelsConfig] = None,
            max_retries: int = 3,
            retry_delay: float = 2.0
    ):
        self._models_config = config or get_models_config()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.metrics = AgentMetrics()

        if llm_provider:
            self._llm = llm_provider
        else:
            llm_config = self._models_config.get_llm_config(self.agent_name)
            self._llm = LLMProviderFactory.create(llm_config)

        provider_value = self._get_provider_value()
        logger.info(
            f"[INIT] {self.__class__.__name__}: "
            f"task={self.task_type.value}, provider={provider_value}"
        )

    def _get_provider_value(self) -> str:
        provider = self._llm.config.provider
        if isinstance(provider, LLMProviderType):
            return provider.value
        return str(provider)

    @property
    def config(self) -> ModelsConfig:
        return self._models_config

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def model(self) -> str:
        return self._llm.config.model

    def get_context_limit(self) -> int:
        """Получить примерный лимит контекста для текущей модели."""
        model = self.model.lower()
        
        if '405b' in model:
            return self.MODEL_CONTEXT_LIMITS['llama-405b']
        elif '235b' in model or 'qwen3' in model:
            return self.MODEL_CONTEXT_LIMITS['qwen-235b']
        elif '70b' in model:
            return self.MODEL_CONTEXT_LIMITS['llama-70b']
        elif 'gemini' in model:
            return self.MODEL_CONTEXT_LIMITS['gemini']
        
        return self.MODEL_CONTEXT_LIMITS['default']

    def check_content_length(self, content: str, max_output_tokens: int = 4096) -> bool:
        """
        Проверить поместится ли контент в контекст.
        
        Args:
            content: Текст для проверки
            max_output_tokens: Ожидаемое количество выходных токенов
            
        Returns:
            True если контент поместится
        """
        # Примерная оценка: 4 символа = 1 токен для русского
        estimated_tokens = len(content) / 4
        limit = self.get_context_limit() / 4  # В токенах
        
        # Оставляем место для системного промпта и ответа
        available = limit - 1000 - max_output_tokens
        
        return estimated_tokens < available

    def _is_context_error(self, error: Exception) -> bool:
        """Проверить является ли ошибка превышением контекста."""
        error_str = str(error).lower()
        return any(phrase in error_str for phrase in [
            'context length',
            'maximum context',
            'token limit',
            'too long',
            'reduce the length'
        ])

    def _parse_context_error(self, error: Exception) -> tuple[int, int]:
        """Извлечь числа из ошибки контекста (requested, limit)."""
        error_str = str(error)
        numbers = re.findall(r'\d+', error_str)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        return 0, 0

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            min_response_length: Optional[int] = None
    ) -> str:
        """
        Генерация текста с retry и обработкой ошибок.
        
        Raises:
            ContextLimitError: Если контент слишком длинный
        """
        self.metrics.total_calls += 1
        start_time = time.time()
        min_len = min_response_length or self.MIN_RESPONSE_LENGTH

        last_error = None
        last_response = ""

        for attempt in range(self.max_retries + 1):
            try:
                if hasattr(self._llm, 'generate_for_task'):
                    result = self._llm.generate_for_task(
                        prompt=prompt,
                        task_type=self.task_type,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                else:
                    result = self._llm.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                # Проверка пустого ответа
                if not result or len(result.strip()) < min_len:
                    self.metrics.empty_responses += 1
                    last_response = result or ""
                    
                    logger.warning(
                        f"[{self.task_type.value}] {self.agent_name}: "
                        f"empty/short response ({len(last_response)} chars), "
                        f"attempt {attempt + 1}/{self.max_retries + 1}"
                    )
                    
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (attempt + 1) * 1.5
                        logger.info(f"[{self.agent_name}] Waiting {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"[{self.agent_name}] All retries exhausted")
                        return last_response

                # Успех
                latency = (time.time() - start_time) * 1000
                self.metrics.successful_calls += 1
                self.metrics.total_latency_ms += latency
                return result

            except Exception as e:
                last_error = e
                
                # Ошибка контекста - не retry, сразу выбрасываем
                if self._is_context_error(e):
                    self.metrics.context_errors += 1
                    self.metrics.failed_calls += 1
                    requested, limit = self._parse_context_error(e)
                    logger.error(
                        f"[{self.agent_name}] Context limit exceeded: "
                        f"requested={requested}, limit={limit}"
                    )
                    raise ContextLimitError(str(e), requested, limit)

                logger.warning(
                    f"[{self.task_type.value}] {self.agent_name} "
                    f"error attempt {attempt + 1}: {e}"
                )

                if attempt < self.max_retries:
                    delay = self.retry_delay * (attempt + 1) * 2
                    time.sleep(delay)

        self.metrics.failed_calls += 1

        if last_error:
            raise last_error

        return last_response

    def generate_structured(
            self,
            prompt: str,
            output_schema: Type[T],
            system_prompt: Optional[str] = None
    ) -> T:
        """Структурированная генерация."""
        self.metrics.total_calls += 1
        start_time = time.time()

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self._llm.generate_structured(
                    prompt=prompt,
                    output_schema=output_schema,
                    system_prompt=system_prompt
                )

                latency = (time.time() - start_time) * 1000
                self.metrics.successful_calls += 1
                self.metrics.total_latency_ms += latency
                return result

            except Exception as e:
                last_error = e
                
                if self._is_context_error(e):
                    self.metrics.context_errors += 1
                    self.metrics.failed_calls += 1
                    raise ContextLimitError(str(e))

                if attempt < self.max_retries:
                    delay = self.retry_delay * (attempt + 1) * 2
                    time.sleep(delay)

        self.metrics.failed_calls += 1
        raise last_error

    def get_metrics(self) -> dict:
        """Метрики агента."""
        return {
            "agent": self.agent_name,
            "task_type": self.task_type.value,
            "model": self.model,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "empty_responses": self.metrics.empty_responses,
            "context_errors": self.metrics.context_errors,
            "success_rate": f"{self.metrics.success_rate:.2%}",
            "avg_latency_ms": f"{self.metrics.avg_latency_ms:.0f}",
        }

    def reset_metrics(self):
        """Сбросить метрики."""
        self.metrics = AgentMetrics()

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Главный метод - реализуется в подклассах."""
        pass
