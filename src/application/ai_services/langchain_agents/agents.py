# -*- coding: utf-8 -*-
"""
LangChain-версии основных текстовых агентов.

Промпты и Pydantic-схемы результата переиспользуются из legacy-агентов —
меняется только транспорт (LangChain ChatModel вместо прямых HTTP-запросов).
Публичный API каждого агента идентичен legacy-версии, поэтому оркестратор
может переключать backend без изменения кода (env AGENT_BACKEND=langchain).
"""

import logging
from typing import Optional

from src.infrastructure.ai.llm_provider import TaskType
from src.application.ai_services.langchain_agents.base import LangChainAgent

# Переиспользуем промпты и схемы legacy-агентов — единый источник правды
from src.application.ai_services.agents.classifier_agent import (
    ClassifierAgent,
    ClassificationResult,
)
from src.application.ai_services.agents.relevance_agent import (
    RelevanceAgent,
    RelevanceResult,
)
from src.application.ai_services.agents.summarizer_agent import (
    SummarizerAgent,
    SummaryResult,
)
from src.application.ai_services.agents.rewriter_agent import (
    RewriterAgent,
    TitleResult,
)

logger = logging.getLogger(__name__)


class LCClassifierAgent(LangChainAgent):
    """Классификация НОВОСТЬ/СТАТЬЯ через LangChain."""

    agent_name = "classifier"
    task_type = TaskType.LIGHT
    MIN_RESPONSE_LENGTH = 5

    def classify(self, title: str, content: str) -> bool:
        return self.process(title, content)

    def classify_with_details(self, title: str, content: str) -> ClassificationResult:
        prompt = ClassifierAgent.CLASSIFICATION_PROMPT.format(
            title=title, content=content[:800]
        )
        try:
            result = self.generate_structured(
                prompt=prompt,
                output_schema=ClassificationResult,
                system_prompt=ClassifierAgent.SYSTEM_PROMPT,
            )
            logger.info(
                f"[LC:Classifier] {'НОВОСТЬ' if result.is_news else 'СТАТЬЯ'} "
                f"({result.confidence:.0%})"
            )
            return result
        except Exception as e:
            logger.error(f"[LC:Classifier] Structured failed: {e}")
            return ClassificationResult(
                is_news=False, confidence=0.5, reasoning="Fallback классификация"
            )

    def process(self, title: str, content: str) -> bool:
        return self.classify_with_details(title, content).is_news


class LCRelevanceAgent(LangChainAgent):
    """Оценка релевантности через LangChain."""

    agent_name = "relevance"
    task_type = TaskType.LIGHT
    MIN_RESPONSE_LENGTH = 5

    def score(self, title: str, content: str, tags: Optional[list[str]] = None) -> tuple[int, str]:
        result = self.process(title, content, tags)
        return (result.score, result.reason)

    def score_with_details(
            self, title: str, content: str, tags: Optional[list[str]] = None
    ) -> RelevanceResult:
        return self.process(title, content, tags)

    def process(
            self, title: str, content: str, tags: Optional[list[str]] = None
    ) -> RelevanceResult:
        tags_str = ", ".join(tags[:5]) if tags else "нет тегов"
        prompt = RelevanceAgent.SCORING_PROMPT.format(
            title=title, tags=tags_str, content=content[:600]
        )
        try:
            result = self.generate_structured(
                prompt=prompt,
                output_schema=RelevanceResult,
                system_prompt=RelevanceAgent.SYSTEM_PROMPT,
            )
            logger.info(f"[LC:Relevance] {result.score}/10 - {result.reason[:50]}...")
            return result
        except Exception as e:
            logger.error(f"[LC:Relevance] failed: {e}")
            return RelevanceResult(
                score=5,
                reason="Ошибка оценки, значение по умолчанию",
                categories=[],
                target_audience="general",
            )


class LCSummarizerAgent(LangChainAgent):
    """Создание тизеров через LangChain."""

    agent_name = "summarizer"
    task_type = TaskType.MEDIUM
    MIN_RESPONSE_LENGTH = 50

    def summarize(self, title: str, content: str) -> str:
        return self.process(title, content).teaser

    def summarize_with_details(self, title: str, content: str) -> SummaryResult:
        return self.process(title, content)

    def process(self, title: str, content: str) -> SummaryResult:
        prompt = SummarizerAgent.SUMMARY_PROMPT.format(
            title=title, content=content[:1200]
        )
        try:
            result = self.generate_structured(
                prompt=prompt,
                output_schema=SummaryResult,
                system_prompt=SummarizerAgent.SYSTEM_PROMPT,
            )
            # Переиспользуем проверенную очистку тизера из legacy-агента
            result.teaser = SummarizerAgent._clean_teaser(self, result.teaser)
            logger.info(f"[LC:Summarizer] Teaser: {len(result.teaser)} chars")
            return result
        except Exception as e:
            logger.error(f"[LC:Summarizer] failed: {e}")
            sentences = content.split(".")[:2]
            fallback = ". ".join(s.strip() for s in sentences if s.strip()) + "."
            return SummaryResult(teaser=fallback[:300], key_points=[], main_topic="")


class LCRewriterAgent(LangChainAgent):
    """Улучшение заголовков через LangChain."""

    agent_name = "rewriter"
    task_type = TaskType.MEDIUM
    MIN_RESPONSE_LENGTH = 15

    def rewrite_title(self, title: str, content: str = "") -> str:
        return self.process(title, content).improved_title

    def rewrite_with_details(
            self, title: str, content: str = "", max_length: int = 100
    ) -> TitleResult:
        result = self.process(title, content)
        result.original_length = len(title)
        result.new_length = len(result.improved_title)
        return result

    def process(self, title: str, content: str = "") -> TitleResult:
        prompt = RewriterAgent.REWRITE_PROMPT.format(title=title)
        try:
            response = self.generate(
                prompt=prompt,
                system_prompt=RewriterAgent.SYSTEM_PROMPT,
                max_tokens=80,
                min_response_length=10,
            )
            # Переиспользуем извлечение/валидацию заголовка из legacy-агента
            improved = RewriterAgent._extract_title(self, response, title)
            improved = RewriterAgent._validate_title(self, improved, title)

            if improved and improved != title and len(improved) >= 15:
                logger.info(f"[LC:Rewriter] '{title[:30]}...' -> '{improved[:30]}...'")
                return TitleResult(
                    improved_title=improved,
                    improvements_made=["Улучшена структура"],
                )

            return TitleResult(
                improved_title=title,
                original_issues=["Заголовок не был улучшен"],
            )
        except Exception as e:
            logger.error(f"[LC:Rewriter] failed: {e}")
            return TitleResult(improved_title=title, original_issues=[str(e)])
