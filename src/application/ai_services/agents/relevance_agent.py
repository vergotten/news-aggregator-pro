# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/relevance_agent.py
# =============================================================================
"""
Расширенный агент оценки релевантности с LangChain.

Оценивает релевантность контента для технической аудитории (разработчики, исследователи, инженеры).
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

from src.application.ai_services.agents.base_agent import BaseAgent
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)


class RelevanceResult(BaseModel):
    """Структурированный вывод для оценки релевантности."""
    
    score: int = Field(
        ge=0, le=10,
        description="Оценка релевантности от 0 до 10"
    )
    reason: str = Field(
        description="Краткое объяснение оценки"
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Релевантные категории (например, 'AI/ML', 'DevOps', 'Security')"
    )
    target_audience: str = Field(
        default="general",
        description="Основная целевая аудитория"
    )


class RelevanceAgent(BaseAgent):
    """
    Агент для оценки релевантности контента для технической аудитории.
    
    Шкала оценки:
    - 9-10: Критически важно (прорыв, важный релиз, проблема безопасности)
    - 7-8: Очень интересно (новые подходы, полезные инструменты, глубокий анализ)
    - 5-6: Умеренно интересно (туториалы, общие темы)
    - 3-4: Низкий интерес (базовые темы, маркетинговый контент)
    - 1-2: Нерелевантно (нетехнический контент)
    - 0: Полностью нерелевантно
    
    Пример:
        >>> agent = RelevanceAgent()
        >>> score, reason = agent.score("Python 3.13 Release", "New JIT compiler...")
        >>> print(score)  # 9
    """
    
    agent_name = "relevance"
    
    SYSTEM_PROMPT = """Ты оценщик релевантности контента для агрегатора технических новостей.
Твоя аудитория: разработчики ПО, DevOps-инженеры, дата-сайентисты, технические исследователи.

Оценивай контент на основе его ценности и интересности для этой технической аудитории.
Будь последовательным и объективным в своих оценках."""
    
    SCORING_PROMPT = """Оцени релевантность этого контента для технической аудитории.

КРИТЕРИИ ОЦЕНКИ:

⭐ 9-10 БАЛЛОВ - КРИТИЧЕСКИ ВАЖНО:
- Прорывное исследование или технология
- Важный релиз популярного инструмента (Python, React, Kubernetes)
- Серьёзная уязвимость или проблема безопасности
- Значительные изменения в индустрии
Примеры: "Python 4.0 Released", "RCE vulnerability in Linux kernel"

⭐ 7-8 БАЛЛОВ - ОЧЕНЬ ИНТЕРЕСНО:
- Новые подходы и методологии
- Полезные инструменты и библиотеки
- Глубокий технический анализ
- Релевантные исследования
Примеры: "New ML architecture", "ORM library comparison"

⭐ 5-6 БАЛЛОВ - УМЕРЕННО ИНТЕРЕСНО:
- Стандартные туториалы и руководства
- Обзоры общих тем
- Личный опыт без уникальных инсайтов
Примеры: "Как настроить Docker", "Мой опыт с React"

⭐ 3-4 БАЛЛОВ - НИЗКИЙ ИНТЕРЕС:
- Базовые темы для начинающих
- Повторение известной информации
- Маркетинговый контент
Примеры: "Что такое переменная", "10 причин выбрать наш продукт"

⭐ 1-2 БАЛЛОВ - НЕРЕЛЕВАНТНО:
- Нетехнический контент
- Реклама без технической ценности
- Спам или низкокачественный контент

⭐ 0 БАЛЛОВ - ПОЛНОСТЬЮ НЕРЕЛЕВАНТНО:
- Политика (если не связана с технологиями)
- Личные истории без технического контекста
- Офтопик

КОНТЕНТ ДЛЯ ОЦЕНКИ:
Заголовок: {title}
Теги: {tags}
Текст (первые 600 символов): {content}

Оцени этот контент и объясни своё обоснование."""
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[ModelsConfig] = None,
        # Обратная совместимость
        ollama_client=None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """Инициализация агента релевантности."""
        if ollama_client is not None:
            logger.warning("ollama_client устарел. Используйте llm_provider.")
        
        super().__init__(llm_provider=llm_provider, config=config)
        logger.info(f"RelevanceAgent инициализирован с моделью: {self.model}")
    
    def score(
        self,
        title: str,
        content: str,
        tags: Optional[list[str]] = None
    ) -> tuple[int, str]:
        """
        Оценка релевантности контента.
        
        Аргументы:
            title: Заголовок контента
            content: Текст контента
            tags: Опциональный список тегов
            
        Возвращает:
            Кортеж (оценка 0-10, причина)
        """
        result = self.process(title, content, tags)
        return (result.score, result.reason)
    
    def score_with_details(
        self,
        title: str,
        content: str,
        tags: Optional[list[str]] = None
    ) -> RelevanceResult:
        """
        Оценка с полными деталями, включая категории.
        
        Аргументы:
            title: Заголовок контента
            content: Текст контента
            tags: Опциональный список тегов
            
        Возвращает:
            RelevanceResult с оценкой, причиной, категориями
        """
        return self.process(title, content, tags)
    
    def process(
        self,
        title: str,
        content: str,
        tags: Optional[list[str]] = None
    ) -> RelevanceResult:
        """
        Главный метод обработки - оценка релевантности контента.
        
        Аргументы:
            title: Заголовок контента
            content: Текст контента
            tags: Опциональный список тегов
            
        Возвращает:
            RelevanceResult
        """
        tags_str = ", ".join(tags[:5]) if tags else "no tags"
        
        prompt = self.SCORING_PROMPT.format(
            title=title,
            tags=tags_str,
            content=content[:600]
        )
        
        try:
            result = self.generate_structured(
                prompt=prompt,
                output_schema=RelevanceResult,
                system_prompt=self.SYSTEM_PROMPT
            )
            
            logger.info(
                f"Оценка релевантности: {result.score}/10 - {result.reason[:50]}..."
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Structured scoring failed: {e}")
            return self._score_simple(title, content, tags_str)
    
    def _score_simple(
        self,
        title: str,
        content: str,
        tags_str: str
    ) -> RelevanceResult:
        """
        Simple fallback scoring without structured output.
        """
        prompt = f"""Rate this content's relevance for tech audience (0-10).

Заголовок: {title}
Теги: {tags_str}
Text: {content[:500]}

Format your response as:
Score: [number 0-10]
Reason: [brief explanation]"""
        
        try:
            response = self.generate(prompt=prompt, max_tokens=200)
            score, reason = self._parse_simple_response(response)
            
            return RelevanceResult(
                score=score,
                reason=reason,
                categories=[],
                target_audience="general"
            )
            
        except Exception as e:
            logger.error(f"Simple scoring failed: {e}")
            return RelevanceResult(
                score=5,
                reason="Scoring error, default value assigned",
                categories=[],
                target_audience="general"
            )
    
    def _parse_simple_response(self, response: str) -> tuple[int, str]:
        """Parse simple response format."""
        import re
        
        score = 5
        reason = "Moderate relevance"
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            
            # Extract score
            if any(kw in line.lower() for kw in ['score:', 'оценка:', 'rating:']):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    score = min(10, max(0, int(numbers[0])))
            
            # Extract reason
            elif any(kw in line.lower() for kw in ['reason:', 'причина:', 'explanation:']):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    reason = parts[1].strip()
        
        return (score, reason)
    
    def filter_by_relevance(
        self,
        элементов: list[tuple[str, str, list[str]]],
        min_score: int = 5
    ) -> list[tuple[int, RelevanceResult]]:
        """
        Filter and score multiple элементов.
        
        Аргументы:
            элементов: List of (title, content, tags) tuples
            min_score: Minimum score to include
            
        Возвращает:
            List of (index, RelevanceResult) for элементов above threshold
        """
        results = []
        
        for i, (title, content, tags) in enumerate(элементов):
            try:
                result = self.process(title, content, tags)
                if result.score >= min_score:
                    results.append((i, result))
            except Exception as e:
                logger.error(f"Scoring error for item {i}: {e}")
        
        # Sort by score descending
        results.sort(key=lambda x: x[1].score, reverse=True)
        return results
