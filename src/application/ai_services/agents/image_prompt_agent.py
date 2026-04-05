# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/image_prompt_agent.py
# =============================================================================
"""
Агент генерации промптов для изображений v2.0

Изменения v2.0:
- Добавлен режим cat_style для генерации промпта с маскотом НейроКотом
- generate_prompt() — стандартный промпт без кота (для обложки)
- generate_cat_prompt() — промпт с котом релевантным теме статьи
- Оба результата сохраняются в article.metadata
"""

import logging
import re
from typing import Optional, List
from pydantic import BaseModel, Field

from src.application.ai_services.agents.base_agent import BaseAgent, TaskType

logger = logging.getLogger(__name__)


class ImagePromptResult(BaseModel):
    """Результат генерации промпта для изображения."""
    prompt: str = Field(description="Промпт для генерации изображения (EN)")
    style: str = Field(default="digital art", description="Стиль изображения")
    success: bool = Field(default=True)


class ImagePromptAgent(BaseAgent):
    """
    Агент генерации промптов для обложек статей v2.0

    Два режима:
    1. generate_prompt() — стандартная обложка без кота
    2. generate_cat_prompt() — обложка с НейроКотом в роли релевантном статье
    """

    agent_name = "image_prompt"
    task_type = TaskType.LIGHT
    MIN_RESPONSE_LENGTH = 20

    # ── Стандартный промпт (без кота) ──
    SYSTEM_PROMPT = """You are an expert at creating image prompts for AI art generators.
Generate a single image prompt in English for a blog post cover image.
Output ONLY the prompt, nothing else. No explanations, no quotes."""

    PROMPT_TEMPLATE = """Create an image prompt for a tech blog article cover.

Article title: {title}
First paragraph: {first_paragraph}
Tags: {tags}

Requirements:
- Prompt in English, 30-80 words
- Style: modern digital illustration, clean, professional
- NO text or letters in the image
- NO people faces (to avoid uncanny valley)
- Use abstract tech concepts, geometric shapes, or symbolic imagery
- Colors: vibrant but professional (blue, teal, purple, orange accents)
- Suitable as a 1200x630 blog header image

IMAGE PROMPT:"""

    # ── Кошачий промпт (с НейроКотом) ──
    CAT_SYSTEM_PROMPT = """You are an expert at creating image prompts for AI art generators.
You create prompts featuring NeuroCat — a specific cat mascot with a defined visual style:
neon glowing outline of a cat's head in profile view, circuit board traces running through 
the body in blue and orange neon lines, microchip on the chest, dark navy background, 
whiskers as thin white neon lines. The cat should have scene-specific elements added 
relevant to the article topic floating around it.
Output ONLY the prompt, nothing else. No explanations, no quotes."""

    CAT_PROMPT_TEMPLATE = """Create an image prompt for NeuroCat blog cover image.

Article title: {title}
Tags: {tags}
Topic: {first_paragraph}

NeuroCat visual style (KEEP EXACTLY):
- Neon glowing outline of cat head in side profile view
- Circuit board PCB traces running through the body in cyan/blue and orange neon lines
- Small microchip icon on the chest area
- Dark deep navy/black background with subtle gradient
- Whiskers as thin bright white neon lines
- Glowing eyes looking forward

Add TOPIC-SPECIFIC floating elements around the cat based on article topic:
- FPGA/hardware/electronics → floating PCB boards, resistors, oscilloscope waves
- AI/LLM/neural networks → floating neural network nodes, glowing synapses, matrix rain
- Security/hacking/privacy → floating lock icons, binary streams, shield symbols  
- Database/data → floating 3D data cubes, SQL brackets, glowing cylinders
- Web/frontend/UI → floating browser windows, CSS brackets, color swatches
- DevOps/cloud/docker → floating containers, cloud shapes, pipeline arrows
- Math/algorithms → floating equations, graph nodes, geometric shapes
- General tech → floating code snippets, gear icons, circuit elements

Requirements:
- English prompt, 50-90 words
- Maintain the neon circuit-cat aesthetic
- Dark background, neon blue + orange color palette
- NO text, NO letters, NO words in the image
- Horizontal 1200x630 format for blog header

CAT IMAGE PROMPT:"""

    def __init__(self, llm_provider=None, config=None, **kwargs):
        super().__init__(
            llm_provider=llm_provider,
            config=config,
            max_retries=2,
            retry_delay=2.0,
        )
        logger.info("[INIT] ImagePromptAgent v2.0 (with NeuroCat mode)")

    def generate_prompt(
        self,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
    ) -> ImagePromptResult:
        """Стандартный промпт для обложки (без кота)."""
        first_paragraph = self._extract_first_paragraph(content, max_chars=500)
        tags_str = ", ".join(tags[:5]) if tags else "technology"

        logger.info(f"[ImagePrompt] Standard prompt for: {title[:50]}...")

        prompt = self.PROMPT_TEMPLATE.format(
            title=title,
            first_paragraph=first_paragraph,
            tags=tags_str,
        )

        try:
            result = self.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.7,
                min_response_length=20,
            )
            cleaned = self._clean_prompt(result)

            if len(cleaned) < 20:
                logger.warning(f"[ImagePrompt] Too short: {len(cleaned)}")
                return self._fallback_prompt(title, tags)

            logger.info(f"[ImagePrompt] OK: {cleaned[:100]}...")
            return ImagePromptResult(prompt=cleaned, style="digital art", success=True)

        except Exception as e:
            logger.warning(f"[ImagePrompt] Error: {e}")
            return self._fallback_prompt(title, tags)

    def generate_cat_prompt(
        self,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
    ) -> ImagePromptResult:
        """
        Промпт с НейроКотом — кот делает что-то релевантное теме статьи.
        Используется для генерации обложки через ChatGPT/DALL-E/FLUX.
        """
        first_paragraph = self._extract_first_paragraph(content, max_chars=300)
        tags_str = ", ".join(tags[:5]) if tags else "technology"

        logger.info(f"[ImagePrompt] Cat prompt for: {title[:50]}...")

        prompt = self.CAT_PROMPT_TEMPLATE.format(
            title=title,
            first_paragraph=first_paragraph,
            tags=tags_str,
        )

        try:
            result = self.generate(
                prompt=prompt,
                system_prompt=self.CAT_SYSTEM_PROMPT,
                max_tokens=250,
                temperature=0.8,  # Чуть больше креативности для кота
                min_response_length=30,
            )
            cleaned = self._clean_prompt(result)

            if len(cleaned) < 30:
                logger.warning(f"[ImagePrompt] Cat prompt too short: {len(cleaned)}")
                return self._fallback_cat_prompt(title, tags)

            logger.info(f"[ImagePrompt] Cat OK: {cleaned[:100]}...")
            return ImagePromptResult(prompt=cleaned, style="cute tech illustration", success=True)

        except Exception as e:
            logger.warning(f"[ImagePrompt] Cat error: {e}")
            return self._fallback_cat_prompt(title, tags)

    def _extract_first_paragraph(self, content: str, max_chars: int = 500) -> str:
        """Извлечь первый абзац из текста."""
        if not content:
            return ""
        lines = content.strip().split("\n")
        paragraphs = []
        current = ""
        for line in lines:
            line = line.strip()
            if not line:
                if current:
                    paragraphs.append(current)
                    current = ""
                continue
            if line.startswith("#"):
                continue
            current = current + " " + line if current else line
        if current:
            paragraphs.append(current)
        if paragraphs:
            return paragraphs[0][:max_chars]
        return content[:max_chars]

    def _clean_prompt(self, text: str) -> str:
        """Очистка промпта от артефактов LLM."""
        if not text:
            return ""
        result = text.strip().strip('"\'')
        prefixes = [
            r'^IMAGE PROMPT:?\s*',
            r'^CAT IMAGE PROMPT:?\s*',
            r'^Prompt:?\s*',
            r'^Here is.*?:\s*',
        ]
        for p in prefixes:
            result = re.sub(p, '', result, flags=re.IGNORECASE)
        result = re.sub(r'\n+', ' ', result)
        result = re.sub(r'\s+', ' ', result)
        return result.strip()

    def _fallback_prompt(self, title: str, tags: Optional[List[str]] = None) -> ImagePromptResult:
        """Fallback стандартный промпт."""
        tag_words = " ".join(tags[:3]) if tags else "technology code"
        prompt = (
            f"Modern digital illustration for tech blog, "
            f"abstract geometric composition representing {tag_words}, "
            f"clean minimalist design, blue and purple gradient, "
            f"floating 3D shapes, glowing connections, "
            f"professional header image, no text, no faces"
        )
        logger.info(f"[ImagePrompt] Fallback standard: {prompt[:80]}...")
        return ImagePromptResult(prompt=prompt, style="digital art", success=False)

    def _fallback_cat_prompt(self, title: str, tags: Optional[List[str]] = None) -> ImagePromptResult:
        """Fallback кошачий промпт."""
        tag_words = " ".join(tags[:2]) if tags else "technology"
        prompt = (
            f"NeuroCat neon circuit cat mascot, cat silhouette made of glowing "
            f"blue cyan and orange neon circuit board traces representing {tag_words}, "
            f"small CPU chip embedded in design, dark navy background, "
            f"sparkling star accent, no text, 1200x630 horizontal blog header"
        )
        logger.info(f"[ImagePrompt] Fallback cat: {prompt[:80]}...")
        return ImagePromptResult(prompt=prompt, style="cute tech illustration", success=False)

    def process(self, title: str, content: str, tags: List[str] = None) -> ImagePromptResult:
        """Основной метод для оркестратора (стандартный промпт)."""
        return self.generate_prompt(title, content, tags)