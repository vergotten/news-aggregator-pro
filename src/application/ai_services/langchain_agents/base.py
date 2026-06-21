# -*- coding: utf-8 -*-
"""
Базовый класс LangChain-агентов.

Зеркалит публичный API legacy BaseAgent (generate / generate_structured /
process / get_metrics), но работает через LangChain ChatModel:
- структурированный вывод через with_structured_output (function calling /
  JSON mode провайдера) с fallback на JSON-парсинг
- retry с экспоненциальной задержкой
- те же AgentMetrics, что у legacy-агентов
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from src.application.ai_services.agents.base_agent import AgentMetrics
from src.config.models_config import ModelsConfig, get_models_config
from src.infrastructure.ai.llm_provider import TaskType
from src.application.ai_services.langchain_agents.llm_factory import build_chat_model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _extract_json(text: str) -> Optional[dict]:
    """Извлечь первый JSON-объект из текста (markdown-фенсы, мусор вокруг)."""
    if not text:
        return None

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


class LangChainAgent(ABC):
    """Базовый LangChain-агент с тем же контрактом, что и legacy BaseAgent."""

    agent_name: str = "base"
    task_type: TaskType = TaskType.MEDIUM
    MIN_RESPONSE_LENGTH: int = 10

    def __init__(
            self,
            config: Optional[ModelsConfig] = None,
            max_retries: int = 3,
            retry_delay: float = 2.0,
            **kwargs,
    ):
        self._models_config = config or get_models_config()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.metrics = AgentMetrics()

        self._llm_config = self._models_config.get_llm_config(self.agent_name)
        self._chat = build_chat_model(self._llm_config)

        logger.info(
            f"[INIT] {self.__class__.__name__} (LangChain): "
            f"task={self.task_type.value}, provider={self._llm_config.provider.value}, "
            f"model={self._llm_config.model}"
        )

    @property
    def config(self) -> ModelsConfig:
        return self._models_config

    @property
    def model(self) -> str:
        return self._llm_config.model

    # =========================================================================
    # Генерация
    # =========================================================================

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> list:
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", prompt))
        return messages

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            min_response_length: Optional[int] = None,
    ) -> str:
        """Текстовая генерация с retry. Контракт идентичен BaseAgent.generate."""
        self.metrics.total_calls += 1
        start_time = time.time()
        min_len = min_response_length or self.MIN_RESPONSE_LENGTH

        chat = self._chat
        bind_kwargs = {}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens
        if bind_kwargs:
            chat = chat.bind(**bind_kwargs)

        messages = self._build_messages(prompt, system_prompt)
        last_error: Optional[Exception] = None
        last_response = ""

        for attempt in range(self.max_retries + 1):
            try:
                response = chat.invoke(messages)
                result = response.content if hasattr(response, "content") else str(response)

                if not result or len(result.strip()) < min_len:
                    self.metrics.empty_responses += 1
                    last_response = result or ""
                    logger.warning(
                        f"[LC:{self.agent_name}] empty/short response "
                        f"({len(last_response)} chars), attempt {attempt + 1}"
                    )
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    return last_response

                self.metrics.successful_calls += 1
                self.metrics.total_latency_ms += (time.time() - start_time) * 1000
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"[LC:{self.agent_name}] attempt {attempt + 1} error: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))

        self.metrics.failed_calls += 1
        if last_error:
            raise last_error
        return last_response

    def generate_structured(
            self,
            prompt: str,
            output_schema: Type[T],
            system_prompt: Optional[str] = None,
    ) -> T:
        """
        Структурированная генерация.

        Сначала пробует нативный structured output провайдера
        (function calling / JSON mode), при неудаче — JSON-инструкция в
        промпте и ручной парсинг.
        """
        self.metrics.total_calls += 1
        start_time = time.time()

        # Попытка 1: нативный structured output
        try:
            structured = self._chat.with_structured_output(output_schema)
            result = structured.invoke(self._build_messages(prompt, system_prompt))
            if isinstance(result, output_schema):
                self.metrics.successful_calls += 1
                self.metrics.total_latency_ms += (time.time() - start_time) * 1000
                return result
        except Exception as e:
            logger.debug(f"[LC:{self.agent_name}] native structured output failed: {e}")

        # Попытка 2: JSON-инструкция + парсинг
        schema_hint = json.dumps(
            output_schema.model_json_schema(), ensure_ascii=False, indent=2
        )
        json_prompt = (
            f"{prompt}\n\n"
            f"Ответь ТОЛЬКО валидным JSON-объектом по этой схеме "
            f"(без markdown, без пояснений):\n{schema_hint}"
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.generate(
                    prompt=json_prompt,
                    system_prompt=system_prompt,
                    temperature=0.1,
                )
                data = _extract_json(raw)
                if data is None:
                    raise ValueError("Не найден валидный JSON в ответе")

                result = output_schema(**data)
                self.metrics.successful_calls += 1
                self.metrics.total_latency_ms += (time.time() - start_time) * 1000
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[LC:{self.agent_name}] structured attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))

        self.metrics.failed_calls += 1
        raise ValueError(f"Не удалось получить структурированный ответ: {last_error}")

    # =========================================================================
    # Метрики
    # =========================================================================

    def get_metrics(self) -> dict:
        return {
            "agent": self.agent_name,
            "backend": "langchain",
            "task_type": self.task_type.value,
            "model": self.model,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "empty_responses": self.metrics.empty_responses,
            "success_rate": f"{self.metrics.success_rate:.2%}",
            "avg_latency_ms": f"{self.metrics.avg_latency_ms:.0f}",
        }

    def reset_metrics(self):
        self.metrics = AgentMetrics()

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Главный метод — реализуется в подклассах."""
