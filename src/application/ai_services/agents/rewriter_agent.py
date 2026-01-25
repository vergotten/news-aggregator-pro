# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/rewriter_agent.py
# =============================================================================
"""
Enhanced Title Rewriter Agent with LangChain.

Improves article titles for professional tech audience.
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from src.application.ai_services.agents.base_agent import BaseAgent
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)


class TitleResult(BaseModel):
    """Structured output for title rewriting."""
    
    improved_title: str = Field(
        description="Improved title (40-80 символов)"
    )
    original_issues: list[str] = Field(
        default_factory=list,
        description="Issues found in original title"
    )
    improvements_made: list[str] = Field(
        default_factory=list,
        description="Improvements applied"
    )
    
    @field_validator('improved_title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Clean and validate improved title."""
        # Remove quotes
        v = v.strip().strip('"').strip("'").strip('`')
        # Remove trailing period
        if v.endswith('.'):
            v = v[:-1]
        # Remove exclamation marks
        v = v.replace('!', '')
        # Remove markdown
        v = v.replace('**', '').replace('*', '')
        return v


class RewriterAgent(BaseAgent):
    """
    Агент для improving article titles.
    
    Title requirements:
    - Length: 40-80 символов
    - Информацияrmative and specific
    - Professional tone
    - NO clickbait
    - NO exclamation marks
    - NO questions
    - NO "Top-5", "10 ways"
    - NO "How I...", "My story..."
    - NO emoji
    
    Пример:
        >>> agent = RewriterAgent()
        >>> title = agent.rewrite_title("How I wrote my framework!!!", "...")
        >>> print(title)  # "Building a Custom Framework: Experience and Solutions"
    """
    
    agent_name = "rewriter"
    
    SYSTEM_PROMPT = """You are a professional tech editor.
Improve article titles to be informative, professional, and engaging.

Rules:
- Keep titles 40-80 символов
- Be specific and informative
- No clickbait or sensationalism
- No personal pronouns focus
- No exclamation marks"""
    
    REWRITE_PROMPT = """Improve this article title to be more professional and informative.

TITLE REQUIREMENTS:

LENGTH:
- 40-80 символов (strict!)
- Not shorter than 40 символов
- Not longer than 80 символов

CONTENT:
- Информацияrmative and specific
- Clearly reflects article essence
- Focus on CONTENT, not on author

STYLE:
- Professional tone
- NO clickbait ("incredible", "shocking")
- NO exclamation marks (!!!)
- NO questions
- NO "Top-5", "10 ways"
- NO "How I...", "My story..."
- NO emoji (they're added separately)

FORMAT:
- Plain text
- NO quotes
- NO period at the end

TRANSFORMATIONS:

❌ BAD → ✅ GOOD:

"How I wrote my framework in a week"
→ "Building a Custom Framework: Experience and Solutions"

"My Linux journey after 10 years on Windows"
→ "Windows to Linux Migration: Practical Experience"

"I learned Docker and here's what happened!!!"
→ "Docker in Production: First Steps and Lessons"

"Incredible! GPT-5 amazed everyone!"
→ "GPT-5: New Capabilities and Improvements"

"Top-5 Python 3.13 features that will blow your mind 🔥"
→ "Python 3.13: Key New Features"

EXCELLENT TITLE EXAMPLES:
✅ "OpenAI Introduces GPT-5 with Enhanced Reasoning"
✅ "Python 3.13 Gains 40% Speed with JIT Compiler"
✅ "New Compression Algorithm Outperforms JPEG by 35%"
✅ "Kubernetes 1.30: What's New in Container Management"
✅ "Monolith to Microservices: A Team's Migration Story"

ORIGINAL TITLE: {title}

ARTICLE CONTEXT (first 400 chars): {content}

Rewrite the title following all requirements."""
    
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
        """Инициализация rewriter agent."""
        if ollama_client is not None:
            logger.warning("ollama_client устарел. Используйте llm_provider.")
        
        super().__init__(llm_provider=llm_provider, config=config)
        logger.info(f"RewriterAgent инициализирован с моделью: {self.model}")
    
    def rewrite_title(self, title: str, content: str) -> str:
        """
        Improve article title.
        
        Аргументы:
            title: Original title
            content: Содержание статьи (for context)
            
        Возвращает:
            Improved title (40-80 символов)
        """
        result = self.process(title, content)
        return result.improved_title
    
    def rewrite_with_details(
        self,
        title: str,
        content: str
    ) -> TitleResult:
        """
        Rewrite title with full analysis.
        
        Аргументы:
            title: Original title
            content: Содержание статьи
            
        Возвращает:
            TitleResult with improved_title, issues, improvements
        """
        return self.process(title, content)
    
    def process(self, title: str, content: str) -> TitleResult:
        """
        Main processing method - rewrite title.
        
        Аргументы:
            title: Original title
            content: Содержание статьи
            
        Возвращает:
            TitleResult
        """
        prompt = self.REWRITE_PROMPT.format(
            title=title,
            content=content[:400]
        )
        
        try:
            result = self.generate_structured(
                prompt=prompt,
                output_schema=TitleResult,
                system_prompt=self.SYSTEM_PROMPT
            )
            
            # Валидация and potentially fix title
            result.improved_title = self._validate_title(
                result.improved_title,
                title
            )
            
            logger.info(
                f"Title rewritten: '{title[:30]}...' → "
                f"'{result.improved_title[:30]}...' ({len(result.improved_title)} chars)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Structured rewriting failed: {e}")
            return self._rewrite_simple(title, content)
    
    def _rewrite_simple(self, title: str, content: str) -> TitleResult:
        """Simple fallback title rewriting."""
        prompt = f"""Improve this title. Make it professional, 40-80 символов.
No clickbait, no exclamation marks, no personal pronouns.

Original: {title}

Context: {content[:300]}

Improved title (only the title, no explanation):"""
        
        try:
            response = self.generate(prompt=prompt, max_tokens=100)
            improved = self._validate_title(response, title)
            
            return TitleResult(
                improved_title=improved,
                original_issues=[],
                improvements_made=[]
            )
            
        except Exception as e:
            logger.error(f"Simple rewriting failed: {e}")
            return TitleResult(
                improved_title=title,  # Return original as fallback
                original_issues=["Rewriting failed"],
                improvements_made=[]
            )
    
    def _validate_title(self, improved: str, original: str) -> str:
        """
        Валидация and fix improved title.
        
        Аргументы:
            improved: Improved title from LLM
            original: Original title (fallback)
            
        Возвращает:
            Валидацияd title
        """
        # Clean common prefixes
        prefixes = [
            'Improved title:', 'Заголовок:', 'Заголовок:',
            '**Improved:**', '**Заголовок:**', 'Улучшенный заголовок:'
        ]
        
        for prefix in prefixes:
            if improved.lower().startswith(prefix.lower()):
                improved = improved[len(prefix):].strip()
        
        # Remove quotes and cleanup
        improved = improved.strip().strip('"').strip("'").strip('`')
        
        # Remove trailing punctuation
        if improved.endswith('.'):
            improved = improved[:-1]
        improved = improved.replace('!', '')
        
        # Remove markdown
        improved = improved.replace('**', '').replace('*', '')
        
        # Валидация length
        if len(improved) < 10:
            logger.warning(f"Improved title too short ({len(improved)} chars), using original")
            return original
        
        if len(improved) > 150:
            logger.warning(f"Improved title too long ({len(improved)} chars), truncating")
            improved = improved[:150].rsplit(' ', 1)[0]
        
        # Log length warnings
        if len(improved) < 40:
            logger.info(f"Title shorter than optimal: {len(improved)} chars")
        elif len(improved) > 80:
            logger.info(f"Title longer than optimal: {len(improved)} chars")
        
        return improved
    
    def batch_rewrite(
        self,
        элементов: list[tuple[str, str]]
    ) -> list[TitleResult]:
        """
        Rewrite multiple titles.
        
        Аргументы:
            элементов: Список кортежей (заголовок, контент)
            
        Возвращает:
            List of TitleResult
        """
        results = []
        for title, content in элементов:
            try:
                result = self.process(title, content)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch rewrite error for '{title[:30]}': {e}")
                results.append(TitleResult(
                    improved_title=title,
                    original_issues=[f"Ошибка: {str(e)}"],
                    improvements_made=[]
                ))
        return results
