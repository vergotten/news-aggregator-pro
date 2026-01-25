# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/style_normalizer_agent.py
# =============================================================================
"""
Расширенный агент нормализации стиля с LangChain.

Removes personal voice, maintains objectivity for professional content.
"""

import logging
import re
from typing import Optional
from pydantic import BaseModel, Field

from src.application.ai_services.agents.base_agent import BaseAgent
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)


class NormalizationResult(BaseModel):
    """Структурированный вывод для нормализации стиля."""
    
    normalized_text: str = Field(
        description="Text with personal pronouns removed, objective style"
    )
    changes_made: list[str] = Field(
        default_factory=list,
        description="List of changes applied"
    )
    personal_pronouns_removed: int = Field(
        default=0,
        description="Count of personal pronouns removed"
    )
    length_ratio: float = Field(
        default=1.0,
        description="Ratio of normalized length to original"
    )


class StyleNormalizerAgent(BaseAgent):
    """
    Агент для normalizing article style to objective professional tone.
    
    Main tasks:
    - Remove personal pronouns ("I", "we", "my", "our")
    - Remove greetings and farewells
    - Remove calls to action ("subscribe", "like")
    - Replace subjective statements with objective ones
    - Maintain content length and facts
    
    CRITICAL:
    - max_tokens must be sufficient for full text (8000+ recommended)
    - Text length should be preserved (ratio ~0.9-1.1)
    
    Пример:
        >>> agent = StyleNormalizerAgent()
        >>> normalized = agent.normalize_full_text("I wrote this code...")
        >>> print(normalized)  # "This code was developed..."
    """
    
    agent_name = "style_normalizer"
    
    SYSTEM_PROMPT = """You are a professional editor specializing in converting personal narratives to objective technical content.

Your task is to rewrite text removing all personal pronouns while:
- Preserving ALL facts, data, and technical information
- Maintaining the same length (don't shorten!)
- Keeping professional, neutral tone
- Using passive voice or impersonal constructions"""
    
    NORMALIZATION_PROMPT = """Rewrite this article text, removing all personal references while maintaining objective professional style.

═══════════════════════════════════════════════════════════════════════
STRICT NORMALIZATION RULES
═══════════════════════════════════════════════════════════════════════

1. REMOVE COMPLETELY:
   ❌ Greetings: "Hello", "Hi everyone", "Dear readers"
   ❌ Farewells: "Thanks for reading", "See you", "Looking forward to feedback"
   ❌ Calls to action: "Subscribe", "Like", "Share"
   ❌ Questions to reader: "What do you think?", "Have you tried?"
   ❌ Emoji: remove all emoji
   ❌ Personal stories: "It happened to me...", "Once I..."

2. REPLACE WITH IMPERSONAL FORMS:
   
   Личные местоимения:
   ❌ "I developed an algorithm..."
   ✅ "An algorithm was developed..."
   
   ❌ "We conducted research..."
   ✅ "Research showed..."
   
   ❌ "I managed to solve the problem..."
   ✅ "The problem was solved using..."
   
   Opinions:
   ❌ "I think this is important..."
   ✅ "Analysis shows the importance..."
   
   Author actions:
   ❌ "I wrote code..."
   ✅ "The code was implemented..."
   
   Direct address:
   ❌ "You can use..."
   ✅ "It's possible to use..."

3. MUST PRESERVE:
   ✅ ALL facts, data, numbers
   ✅ ALL technical terms exactly as in original
   ✅ ALL code examples (if any)
   ✅ Structure and logic
   ✅ Text length (DO NOT shorten!)

═══════════════════════════════════════════════════════════════════════
КРИТИЧЕСКИ ВАЖНО: PRESERVE FULL TEXT VOLUME!
═══════════════════════════════════════════════════════════════════════

TEXT TO PROCESS:

{content}

REWRITTEN TEXT (completely, preserving volume):"""
    
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
        """Инициализация style normalizer agent."""
        if ollama_client is not None:
            logger.warning("ollama_client устарел. Используйте llm_provider.")
        
        super().__init__(llm_provider=llm_provider, config=config)
        
        # Warn if max_tokens seems low
        if self._llm.config.max_tokens < 4000:
            logger.warning(
                f"max_tokens={self._llm.config.max_tokens} may be insufficient "
                f"for long articles! Recommended: 8000+"
            )
        
        logger.info(
            f"StyleNormalizerAgent initialized: model={self.model}, "
            f"max_tokens={self._llm.config.max_tokens}"
        )
    
    def normalize_full_text(self, content: str) -> str:
        """
        Normalize FULL article text.
        
        Аргументы:
            content: Full article text
            
        Возвращает:
            Нормализованный текст (approximately same length)
        """
        result = self.process(content)
        return result.normalized_text
    
    def normalize_with_details(self, content: str) -> NormalizationResult:
        """
        Normalize with full analysis details.
        
        Аргументы:
            content: Full article text
            
        Возвращает:
            NormalizationResult with normalized_text, changes, metrics
        """
        return self.process(content)
    
    def process(self, content: str) -> NormalizationResult:
        """
        Main processing method - normalize article style.
        
        Аргументы:
            content: Содержание статьи
            
        Возвращает:
            NormalizationResult
        """
        if not content or len(content) < 100:
            logger.warning("Content too short for normalization")
            return NormalizationResult(
                normalized_text=content,
                changes_made=["Content too short, skipped"],
                length_ratio=1.0
            )
        
        original_length = len(content)
        logger.info(f"Normalizing text: {original_length} chars")
        
        prompt = self.NORMALIZATION_PROMPT.format(content=content)
        
        try:
            normalized = self.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT
            )
            
            normalized = normalized.strip()
            
            # Валидация result
            result_length = len(normalized)
            ratio = result_length / original_length if original_length > 0 else 0
            
            logger.info(
                f"Normalization complete: {original_length} → {result_length} chars "
                f"(ratio: {ratio:.2f})"
            )
            
            # Quality checks
            if result_length < 100:
                logger.error("Нормализованный текст too short! Returning original")
                return NormalizationResult(
                    normalized_text=content,
                    changes_made=["Normalization failed - result too short"],
                    length_ratio=1.0
                )
            
            if ratio < 0.5:
                logger.warning(f"Text significantly shortened! ratio={ratio:.2f}")
            
            if ratio > 1.5:
                logger.warning(f"Text significantly expanded! ratio={ratio:.2f}")
            
            # Count removed pronouns
            pronouns_removed = self._count_pronouns_diff(content, normalized)
            
            return NormalizationResult(
                normalized_text=normalized,
                changes_made=["Личные местоимения replaced", "Objective style applied"],
                personal_pronouns_removed=pronouns_removed,
                length_ratio=ratio
            )
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}", exc_info=True)
            return NormalizationResult(
                normalized_text=content,
                changes_made=[f"Ошибка: {str(e)}"],
                length_ratio=1.0
            )
    
    def normalize_intro(self, content: str) -> str:
        """
        Normalize only the introduction (faster processing).
        
        Аргументы:
            content: Full text
            
        Возвращает:
            Text with normalized introduction
        """
        intro_length = min(1500, len(content))
        intro = content[:intro_length]
        
        prompt = f"""Rewrite this article introduction, removing personal references.

REMOVE: "Hello", "I [do something]", "Want to tell", "Looking forward to"
REPLACE with impersonal forms.

ORIGINAL:
{intro}

RESULT:"""
        
        try:
            response = self.generate(prompt=prompt, max_tokens=1000)
            normalized_intro = response.strip()
            
            if len(normalized_intro) < 50:
                return content
            
            if len(content) > intro_length:
                return normalized_intro + "\n\n" + content[intro_length:]
            return normalized_intro
                
        except Exception as e:
            logger.error(f"Intro normalization failed: {e}")
            return content
    
    def _count_pronouns_diff(self, original: str, normalized: str) -> int:
        """Count approximate difference in personal pronouns."""
        patterns = [
            r'\bя\b', r'\bмы\b', r'\bмой\b', r'\bмоя\b', r'\bмоё\b', r'\bмои\b',
            r'\bнаш\b', r'\bнаша\b', r'\bнаше\b', r'\bнаши\b',
            r'\bмне\b', r'\bнам\b', r'\bменя\b', r'\bнас\b',
            r'\bI\b', r'\bwe\b', r'\bmy\b', r'\bour\b', r'\bme\b', r'\bus\b',
        ]
        
        def count_pronouns(text: str) -> int:
            count = 0
            text_lower = text.lower()
            for pattern in patterns:
                count += len(re.findall(pattern, text_lower, re.IGNORECASE))
            return count
        
        original_count = count_pronouns(original)
        normalized_count = count_pronouns(normalized)
        
        return max(0, original_count - normalized_count)
    
    def validate_normalization(self, original: str, normalized: str) -> dict:
        """
        Валидация normalization quality.
        
        Аргументы:
            original: Оригинальный текст
            normalized: Нормализованный текст
            
        Возвращает:
            Validation results dict
        """
        original_len = len(original)
        normalized_len = len(normalized)
        ratio = normalized_len / original_len if original_len > 0 else 0
        
        patterns = [r'\bя\b', r'\bмы\b', r'\bмой\b', r'\bнаш\b',
                    r'\bI\b', r'\bwe\b', r'\bmy\b', r'\bour\b']
        
        remaining_pronouns = 0
        normalized_lower = normalized.lower()
        for pattern in patterns:
            remaining_pronouns += len(re.findall(pattern, normalized_lower))
        
        is_valid = (ratio >= 0.7 and ratio <= 1.3 and remaining_pronouns <= 5)
        
        issues = []
        if ratio < 0.7:
            issues.append(f"Text shortened (ratio: {ratio:.2f})")
        if ratio > 1.3:
            issues.append(f"Text expanded (ratio: {ratio:.2f})")
        if remaining_pronouns > 5:
            issues.append(f"Pronouns remain ({remaining_pronouns})")
        
        return {
            "is_valid": is_valid,
            "length_ratio": ratio,
            "original_length": original_len,
            "normalized_length": normalized_len,
            "remaining_pronouns": remaining_pronouns,
            "issues": issues
        }
