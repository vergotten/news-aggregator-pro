# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/base_agent.py
# =============================================================================
"""
Базовый класс для всех AI агентов.

Предоставляет:
- Интеграцию с LLM провайдером
- Структурированный вывод через Pydantic
- Автоматические retry при ошибках
- Метрики производительности
- Логирование

Использование:
    class MyAgent(BaseAgent):
        agent_name = "my_agent"
        
        def process(self, input_data):
            return self.generate("Обработай: " + input_data)
"""

from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar, Any
import logging
import time
from dataclasses import dataclass, field

from pydantic import BaseModel

from src.infrastructure.ai.llm_provider import LLMProvider, LLMProviderFactory
from src.config.models_config import ModelsConfig, get_models_config

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


@dataclass
class AgentMetrics:
    """Метрики производительности агента."""
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Вычислить процент успешных вызовов."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def avg_latency_ms(self) -> float:
        """Вычислить среднюю задержку."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency_ms / self.successful_calls


class BaseAgent(ABC):
    """
    Абстрактный базовый класс для всех AI агентов.
    
    Предоставляет:
    - Единый доступ к LLM провайдеру
    - Структурированный вывод с Pydantic
    - Автоматические повторы при ошибках
    - Метрики производительности
    - Логирование
    
    Подклассы должны определить:
    - agent_name: str - имя агента для конфигурации
    - process(): абстрактный метод обработки
    """
    
    # Подклассы должны переопределить
    agent_name: str = "base"
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[ModelsConfig] = None,
        max_retries: int = 2,
        retry_delay: float = 1.0
    ):
        """
        Инициализация базового агента.
        
        Аргументы:
            llm_provider: Готовый LLM провайдер (опционально)
            config: Конфигурация моделей (опционально, иначе глобальная)
            max_retries: Максимум повторов при ошибке
            retry_delay: Задержка между повторами в секундах
        """
        self.config = config or get_models_config()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.metrics = AgentMetrics()
        
        # Инициализация LLM провайдера
        if llm_provider:
            self._llm = llm_provider
        else:
            llm_config = self.config.get_llm_config(self.agent_name)
            self._llm = LLMProviderFactory.create(llm_config)
        
        logger.info(
            f"{self.__class__.__name__} инициализирован: "
            f"model={self._llm.config.model}, "
            f"provider={self._llm.config.provider.value}"
        )
    
    @property
    def llm(self) -> LLMProvider:
        """Получить LLM провайдер."""
        return self._llm
    
    @property
    def model(self) -> str:
        """Получить имя модели (для обратной совместимости)."""
        return self._llm.config.model
    
    @property
    def temperature(self) -> float:
        """Получить температуру (для обратной совместимости)."""
        return self._llm.config.temperature
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Сгенерировать текст с автоматическими retry и метриками.
        
        Аргументы:
            prompt: Промпт пользователя
            system_prompt: Системная инструкция (опционально)
            temperature: Переопределить температуру
            max_tokens: Переопределить max_tokens
            
        Возвращает:
            Сгенерированный текст
        """
        self.metrics.total_calls += 1
        start_time = time.time()
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self._llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Записать метрики успеха
                latency = (time.time() - start_time) * 1000
                self.metrics.successful_calls += 1
                self.metrics.total_latency_ms += latency
                
                logger.debug(
                    f"{self.agent_name} сгенерировал ответ: "
                    f"{len(result)} символов за {latency:.0f}мс"
                )
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{self.agent_name} попытка {attempt + 1}/{self.max_retries + 1} "
                    f"не удалась: {e}"
                )
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        # Все попытки провалились
        self.metrics.failed_calls += 1
        logger.error(
            f"{self.agent_name} не удалось после {self.max_retries + 1} попыток"
        )
        raise last_error
    
    def generate_structured(
        self,
        prompt: str,
        output_schema: Type[T],
        system_prompt: Optional[str] = None
    ) -> T:
        """
        Сгенерировать структурированный вывод через Pydantic схему.
        
        Аргументы:
            prompt: Промпт пользователя
            output_schema: Pydantic модель
            system_prompt: Системная инструкция
            
        Возвращает:
            Распарсенный Pydantic объект
        """
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
                
                # Записать метрики успеха
                latency = (time.time() - start_time) * 1000
                self.metrics.successful_calls += 1
                self.metrics.total_latency_ms += latency
                
                logger.debug(
                    f"{self.agent_name} структурированный ответ за {latency:.0f}мс"
                )
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{self.agent_name} структурированная попытка {attempt + 1} "
                    f"не удалась: {e}"
                )
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        # Все попытки провалились
        self.metrics.failed_calls += 1
        raise last_error
    
    def get_metrics(self) -> dict:
        """Получить метрики производительности агента."""
        return {
            "agent": self.agent_name,
            "model": self.model,
            "provider": self._llm.config.provider.value,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "success_rate": f"{self.metrics.success_rate:.2%}",
            "avg_latency_ms": f"{self.metrics.avg_latency_ms:.0f}",
        }
    
    def reset_metrics(self):
        """Сбросить метрики производительности."""
        self.metrics = AgentMetrics()
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Главный метод обработки - должен быть реализован в подклассах.
        
        Это основной интерфейс для использования агента.
        """
        pass
