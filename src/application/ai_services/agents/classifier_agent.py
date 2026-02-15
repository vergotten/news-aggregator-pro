# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/classifier_agent.py
# =============================================================================
"""
Агент классификации контента v8.0

Изменения v8:
- Улучшенная обработка ошибок
- Исключение проблемных моделей
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

from src.application.ai_services.agents.base_agent import BaseAgent, TaskType
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)


class ClassificationResult(BaseModel):
    """Результат классификации."""
    
    is_news: bool = Field(description="True если НОВОСТЬ, False если СТАТЬЯ")
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность 0.0-1.0")
    reasoning: str = Field(description="Обоснование")


class ClassifierAgent(BaseAgent):
    """Агент классификации: НОВОСТЬ или СТАТЬЯ."""
    
    agent_name = "classifier"
    task_type = TaskType.LIGHT
    MIN_RESPONSE_LENGTH = 5
    
    EXCLUDED_MODELS = [
        "openai/gpt-oss-120b:free",
    ]
    
    SYSTEM_PROMPT = """Ты - классификатор контента для технического портала.
Определяй тип: НОВОСТЬ или СТАТЬЯ.
Отвечай на русском."""

    CLASSIFICATION_PROMPT = """Классифицируй контент как НОВОСТЬ или СТАТЬЯ.

НОВОСТЬ:
- Сообщает о конкретном недавнем событии
- Анонс релиза, конференции, исследования
- Короткий формат (< 3000 символов)
- Фокус: ЧТО ПРОИЗОШЛО

СТАТЬЯ:
- Подробный разбор, туториал
- Личный опыт, кейс-стади
- Длинный формат (> 3000 символов)
- Фокус: КАК СДЕЛАТЬ

Заголовок: {title}

Текст (800 символов): {content}

Классифицируй и объясни."""
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[ModelsConfig] = None,
        **kwargs
    ):
        super().__init__(llm_provider=llm_provider, config=config)
        logger.info(f"[INIT] ClassifierAgent v8: model={self.model}")
    
    def classify(self, title: str, content: str) -> bool:
        """Классифицировать -> True=НОВОСТЬ, False=СТАТЬЯ."""
        return self.process(title, content)
    
    def classify_with_details(self, title: str, content: str) -> ClassificationResult:
        """Классифицировать с деталями."""
        prompt = self.CLASSIFICATION_PROMPT.format(
            title=title,
            content=content[:800]
        )
        
        try:
            result = self.generate_structured(
                prompt=prompt,
                output_schema=ClassificationResult,
                system_prompt=self.SYSTEM_PROMPT
            )
            
            logger.info(
                f"[Classifier] {'НОВОСТЬ' if result.is_news else 'СТАТЬЯ'} "
                f"(confidence: {result.confidence:.0%})"
            )
            return result
            
        except Exception as e:
            logger.error(f"[Classifier] Structured failed: {e}")
            is_news = self._classify_simple(title, content)
            return ClassificationResult(
                is_news=is_news,
                confidence=0.5,
                reasoning="Fallback классификация"
            )
    
    def process(self, title: str, content: str) -> bool:
        """Главный метод."""
        try:
            result = self.classify_with_details(title, content)
            return result.is_news
        except Exception as e:
            logger.error(f"[Classifier] Error: {e}")
            return self._classify_simple(title, content)
    
    def _classify_simple(self, title: str, content: str) -> bool:
        """Простая fallback классификация."""
        prompt = f"""Классифицируй. Ответь ОДНИМ словом: НОВОСТЬ или СТАТЬЯ

Заголовок: {title}
Текст: {content[:600]}

Ответ:"""
        
        try:
            response = self.generate(
                prompt=prompt,
                system_prompt="Отвечай одним словом: НОВОСТЬ или СТАТЬЯ",
                max_tokens=10,
                min_response_length=3
            )
            
            response_upper = response.upper().strip()
            is_news = "НОВОСТЬ" in response_upper or "NEWS" in response_upper
            
            logger.info(f"[Classifier] Simple: {'НОВОСТЬ' if is_news else 'СТАТЬЯ'}")
            return is_news
            
        except Exception as e:
            logger.error(f"[Classifier] Simple failed: {e}")
            return False  # Default: СТАТЬЯ
    
    def batch_classify(self, items: list[tuple[str, str]]) -> list[ClassificationResult]:
        """Batch классификация."""
        results = []
        for title, content in items:
            try:
                result = self.classify_with_details(title, content)
                results.append(result)
            except Exception as e:
                results.append(ClassificationResult(
                    is_news=False,
                    confidence=0.0,
                    reasoning=f"Error: {e}"
                ))
        return results
