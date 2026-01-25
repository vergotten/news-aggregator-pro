# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/summarizer_agent.py
# =============================================================================
"""
Enhanced Content Summarizer Agent with LangChain.

Creates teasers and summaries for articles with quality control.
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from src.application.ai_services.agents.base_agent import BaseAgent
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)


class SummaryResult(BaseModel):
    """Structured output for summary generation."""
    
    teaser: str = Field(
        description="Short teaser (2-4 sentences, 150-300 chars)"
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="List of 2-4 key points from the content"
    )
    main_topic: str = Field(
        default="",
        description="Main topic in 2-3 слов"
    )
    
    @field_validator('teaser')
    @classmethod
    def validate_teaser_length(cls, v: str) -> str:
        """Ensure teaser is within reasonable length."""
        if len(v) < 50:
            raise ValueError("Teaser too short (min 50 chars)")
        if len(v) > 500:
            # Truncate to last complete sentence
            v = v[:500]
            last_period = v.rfind('.')
            if last_period > 300:
                v = v[:last_period + 1]
        return v


class SummarizerAgent(BaseAgent):
    """
    Агент для creating content summaries and teasers.
    
    Генерацияs:
    - Teasers: 2-4 sentences (150-300 chars) for feed preview
    - Ключевые пункты: Main takeaways
    - Topic identification
    
    Quality requirements:
    - Информацияrmative and neutral tone
    - NO personal pronouns ("I", "we", "my")
    - NO clickbait ("you won't believe", "shocking")
    - NO questions to reader
    - Facts and specifics, not abstractions
    
    Пример:
        >>> agent = SummarizerAgent()
        >>> summary = agent.summarize("Python 3.13", "New JIT compiler...")
        >>> print(summary)  # "Python 3.13 introduces JIT compilation..."
    """
    
    agent_name = "summarizer"
    
    SYSTEM_PROMPT = """You are a professional tech content summarizer.
Create concise, informative summaries for a tech news feed.

Rules:
- Be factual and objective
- No personal pronouns (I, we, my)
- No clickbait or sensationalism
- No questions to the reader
- Include specific facts and numbers когда доступно"""
    
    SUMMARY_PROMPT = """Create a teaser for this article for a news feed preview.

TEASER REQUIREMENTS:

LENGTH:
- 2-4 sentences
- 150-300 символов
- Enough to understand the essence, creates interest

CONTENT:
- Main idea of the article
- Key fact or number (if present)
- Specifics, not abstractions

STYLE:
- Информацияrmative and neutral
- NO personal pronouns ("I", "we", "my")
- NO evaluative слов ("interesting", "cool", "amazing")
- NO clickbait ("you won't believe", "shocking facts")
- NO questions to reader
- NO emoji

FORMAT:
- Plain text, NO lists or headers
- Complete sentences

GOOD EXAMPLES:
✅ "OpenAI team introduced GPT-5 with improved logical reasoning. The new model shows 40% accuracy improvement on math and programming tasks. API access opens in February 2026."

✅ "Python 3.13 received JIT compiler, accelerating code execution by 40% in most scenarios. Changes affected interpreter and memory management system."

BAD EXAMPLES:
❌ "I found an interesting article about GPT-5. This is a very cool model! Details inside."
(Reason: personal pronouns, evaluative слов, no specifics)

❌ "Learn everything about new Python! You won't believe how fast it is!"
(Reason: clickbait, exclamations, no facts)

ARTICLE TO SUMMARIZE:
Заголовок: {title}

Text (first 1200 chars): {content}

Генерация the teaser, key points, and identify the main topic."""
    
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
        """Инициализация summarizer agent."""
        if ollama_client is not None:
            logger.warning("ollama_client устарел. Используйте llm_provider.")
        
        super().__init__(llm_provider=llm_provider, config=config)
        logger.info(f"SummarizerAgent инициализирован с моделью: {self.model}")
    
    def summarize(self, title: str, content: str) -> str:
        """
        Create a teaser for the article.
        
        Аргументы:
            title: Заголовок статьи
            content: Содержание статьи
            
        Возвращает:
            Teaser text (2-4 sentences, ~150-300 chars)
        """
        result = self.process(title, content)
        return result.teaser
    
    def summarize_with_details(
        self,
        title: str,
        content: str
    ) -> SummaryResult:
        """
        Create summary with full details.
        
        Аргументы:
            title: Заголовок статьи
            content: Содержание статьи
            
        Возвращает:
            SummaryResult with teaser, key_points, main_topic
        """
        return self.process(title, content)
    
    def process(self, title: str, content: str) -> SummaryResult:
        """
        Main processing method - create summary.
        
        Аргументы:
            title: Заголовок статьи
            content: Содержание статьи
            
        Возвращает:
            SummaryResult
        """
        prompt = self.SUMMARY_PROMPT.format(
            title=title,
            content=content[:1200]
        )
        
        try:
            result = self.generate_structured(
                prompt=prompt,
                output_schema=SummaryResult,
                system_prompt=self.SYSTEM_PROMPT
            )
            
            # Post-process teaser
            result.teaser = self._clean_teaser(result.teaser)
            
            logger.info(
                f"Summary created: {len(result.teaser)} chars, "
                f"{len(result.key_points)} key points"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Structured summary failed: {e}")
            return self._summarize_simple(title, content)
    
    def _summarize_simple(self, title: str, content: str) -> SummaryResult:
        """Simple fallback summarization."""
        prompt = f"""Create a brief teaser (2-4 sentences) for this article.
Be factual and objective. No personal pronouns.

Заголовок: {title}

Text: {content[:800]}

Teaser:"""
        
        try:
            response = self.generate(prompt=prompt, max_tokens=200)
            teaser = self._clean_teaser(response)
            
            return SummaryResult(
                teaser=teaser,
                key_points=[],
                main_topic=""
            )
            
        except Exception as e:
            logger.error(f"Simple summary failed: {e}")
            # Ultimate fallback: first 2 sentences
            sentences = content.split('.')[:2]
            fallback = '. '.join(s.strip() for s in sentences if s.strip()) + '.'
            
            return SummaryResult(
                teaser=fallback[:300],
                key_points=[],
                main_topic=""
            )
    
    def _clean_teaser(self, teaser: str) -> str:
        """
        Clean and validate teaser text.
        
        Аргументы:
            teaser: Raw teaser text
            
        Возвращает:
            Cleaned teaser
        """
        cleaned = teaser.strip()
        
        # Remove common prefixes
        prefixes = [
            'Teaser:', 'Summary:', 'Тизер:', 'Краткое описание:',
            '**Teaser:**', '**Summary:**', '**Тизер:**'
        ]
        
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove quotes
        cleaned = cleaned.strip('"').strip("'").strip('`')
        
        # Remove markdown bold
        cleaned = cleaned.replace('**', '')
        
        # Валидация length
        if len(cleaned) < 50:
            logger.warning(f"Teaser too short: {len(cleaned)} chars")
        elif len(cleaned) > 500:
            logger.warning(f"Teaser too long: {len(cleaned)} chars, truncating")
            sentences = cleaned.split('.')
            cleaned = '. '.join(sentences[:4]) + '.'
            if len(cleaned) > 500:
                cleaned = cleaned[:497] + '...'
        
        return cleaned
    
    def batch_summarize(
        self,
        элементов: list[tuple[str, str]]
    ) -> list[SummaryResult]:
        """
        Summarize multiple элементов.
        
        Аргументы:
            элементов: Список кортежей (заголовок, контент)
            
        Возвращает:
            List of SummaryResult
        """
        results = []
        for title, content in элементов:
            try:
                result = self.process(title, content)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch summary error for '{title[:30]}': {e}")
                results.append(SummaryResult(
                    teaser=f"Summary unavailable for: {title[:50]}",
                    key_points=[],
                    main_topic=""
                ))
        return results
