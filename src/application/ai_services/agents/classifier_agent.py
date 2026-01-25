# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/classifier_agent.py
# =============================================================================
"""
Агент классификации контента: НОВОСТЬ или СТАТЬЯ.

Использует структурированный вывод для точной классификации.
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

from src.application.ai_services.agents.base_agent import BaseAgent
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)


class ClassificationResult(BaseModel):
    """Структурированный результат классификации."""
    
    is_news: bool = Field(
        description="True если НОВОСТЬ, False если СТАТЬЯ"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Уверенность в классификации (0.0-1.0)"
    )
    reasoning: str = Field(
        description="Краткое обоснование решения"
    )


class ClassifierAgent(BaseAgent):
    """
    Агент классификации: НОВОСТЬ или СТАТЬЯ.
    
    НОВОСТЬ - характеристики:
    - Сообщает о конкретном недавнем событии
    - Содержит даты и актуальную информацию
    - Анонсирует релиз, конференцию, исследование
    - Короткий, фактологичный (обычно < 3000 символов)
    - Фокус на ЧТО ПРОИЗОШЛО
    
    СТАТЬЯ - характеристики:
    - Подробный разбор, туториал, гайд
    - Личный опыт, кейс-стади
    - Длинный и детальный (> 3000 символов)
    - Фокус на КАК ЭТО РАБОТАЕТ
    
    Пример:
        >>> agent = ClassifierAgent()
        >>> is_news = agent.classify("Вышел Python 3.13", "Новая версия...")
        >>> print(is_news)  # True
    """
    
    agent_name = "classifier"
    
    SYSTEM_PROMPT = """Ты классификатор контента для технического новостного агрегатора.
Твоя задача - определить, является ли контент НОВОСТЬЮ или СТАТЬЁЙ.

Будь точным и последовательным в классификации."""
    
    CLASSIFICATION_PROMPT = """Проанализируй контент и классифицируй его как НОВОСТЬ или СТАТЬЯ.

КРИТЕРИИ КЛАССИФИКАЦИИ:

НОВОСТЬ - если контент:
✓ Сообщает о конкретном недавнем событии
✓ Содержит актуальную информацию с датами
✓ Анонсирует релиз, конференцию, публикацию исследования
✓ Короткий и фактологичный (обычно < 3000 символов)
✓ Фокус на ЧТО ПРОИЗОШЛО, а не КАК СДЕЛАТЬ
✓ Примеры: "Вышел Python 3.13", "OpenAI анонсировала GPT-5", "Обнаружена критическая уязвимость"

СТАТЬЯ - если контент:
✓ Подробный разбор темы, туториал, руководство
✓ Личный опыт, кейс-стади, обзор решения
✓ Длинный и детальный (> 3000 символов)
✓ Глубокий анализ без привязки к конкретному событию
✓ Фокус на КАК ЭТО РАБОТАЕТ или КАК ЭТО СДЕЛАТЬ
✓ Примеры: "Как настроить Docker", "Миграция на микросервисы", "Обзор архитектуры"

КОНТЕНТ ДЛЯ АНАЛИЗА:
Заголовок: {title}

Текст (первые 800 символов): {content}

Классифицируй контент и объясни своё решение."""
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[ModelsConfig] = None,
        ollama_client=None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Инициализация агента классификации.
        
        Аргументы:
            llm_provider: LLM провайдер (рекомендуется)
            config: Конфигурация моделей
            ollama_client: Устаревший параметр, для обратной совместимости
            model: Игнорируется, используется конфигурация
            temperature: Игнорируется, используется конфигурация
            max_tokens: Игнорируется, используется конфигурация
        """
        if ollama_client is not None:
            logger.warning(
                "Параметр ollama_client устарел. "
                "Используйте llm_provider или config."
            )
        
        super().__init__(llm_provider=llm_provider, config=config)
        logger.info(f"ClassifierAgent инициализирован с моделью: {self.model}")
    
    def classify(self, title: str, content: str) -> bool:
        """
        Классифицировать контент как НОВОСТЬ или СТАТЬЯ.
        
        Аргументы:
            title: Заголовок
            content: Текст контента
            
        Возвращает:
            True если НОВОСТЬ, False если СТАТЬЯ
        """
        return self.process(title, content)
    
    def classify_with_details(self, title: str, content: str) -> ClassificationResult:
        """
        Классифицировать с полными деталями.
        
        Аргументы:
            title: Заголовок
            content: Текст контента
            
        Возвращает:
            ClassificationResult с is_news, confidence, reasoning
        """
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
                f"Классификация: {'НОВОСТЬ' if result.is_news else 'СТАТЬЯ'} "
                f"(уверенность: {result.confidence:.2f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Структурированная классификация не удалась: {e}")
            is_news = self._classify_simple(title, content)
            return ClassificationResult(
                is_news=is_news,
                confidence=0.5,
                reasoning="Fallback классификация из-за ошибки парсинга"
            )
    
    def process(self, title: str, content: str) -> bool:
        """
        Главный метод обработки - классификация контента.
        
        Аргументы:
            title: Заголовок
            content: Текст контента
            
        Возвращает:
            True если НОВОСТЬ, False если СТАТЬЯ
        """
        try:
            result = self.classify_with_details(title, content)
            return result.is_news
        except Exception as e:
            logger.error(f"Классификация не удалась: {e}", exc_info=True)
            return self._classify_simple(title, content)
    
    def _classify_simple(self, title: str, content: str) -> bool:
        """
        Простая fallback классификация без структурированного вывода.
        
        Аргументы:
            title: Заголовок
            content: Текст контента
            
        Возвращает:
            True если НОВОСТЬ, False если СТАТЬЯ
        """
        prompt = f"""Классифицируй контент. Ответь ТОЛЬКО одним словом: НОВОСТЬ или СТАТЬЯ

Заголовок: {title}
Текст: {content[:600]}

Ответ:"""
        
        try:
            response = self.generate(
                prompt=prompt,
                system_prompt="Отвечай ровно одним словом: НОВОСТЬ или СТАТЬЯ",
                max_tokens=10
            )
            
            response_upper = response.upper().strip()
            is_news = "НОВОСТЬ" in response_upper or "NEWS" in response_upper
            
            logger.info(f"Простая классификация: {'НОВОСТЬ' if is_news else 'СТАТЬЯ'}")
            return is_news
            
        except Exception as e:
            logger.error(f"Простая классификация не удалась: {e}")
            # Крайний fallback: считаем СТАТЬЁЙ (безопаснее)
            return False
    
    def batch_classify(
        self,
        элементов: list[tuple[str, str]]
    ) -> list[ClassificationResult]:
        """
        Классифицировать несколько элементов.
        
        Аргументы:
            элементов: Список кортежей (title, content)
            
        Возвращает:
            Список ClassificationResult
        """
        results = []
        for title, content in элементов:
            try:
                result = self.classify_with_details(title, content)
                results.append(result)
            except Exception as e:
                logger.error(f"Ошибка batch классификации для '{title[:30]}': {e}")
                results.append(ClassificationResult(
                    is_news=False,
                    confidence=0.0,
                    reasoning=f"Ошибка: {str(e)}"
                ))
        return results
