# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/image_prompt_agent.py
# =============================================================================
"""
Агент генерации промптов для изображений v1.0

Шаг 9 в оркестраторе: анализирует первый абзац и теги статьи,
генерирует промпт на английском для создания обложки.

Промпт оптимизирован для text-to-image моделей (FLUX, Stable Diffusion,
Ollama z-image-turbo, DALL-E).

Результат сохраняется в article_metadata['image_prompt'].
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
    Агент генерации промптов для обложек статей v1.0

    Берёт первый абзац + теги → генерирует описание картинки
    на английском для text-to-image моделей.

    Лёгкий агент — один короткий вызов LLM.
    """

    agent_name = "image_prompt"
    task_type = TaskType.LIGHT
    MIN_RESPONSE_LENGTH = 20

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

    def __init__(self, llm_provider=None, config=None, **kwargs):
        super().__init__(
            llm_provider=llm_provider,
            config=config,
            max_retries=2,
            retry_delay=2.0,
        )
        logger.info("[INIT] ImagePromptAgent v1.0")

    def generate_prompt(
        self,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
    ) -> ImagePromptResult:
        """
        Генерирует промпт для обложки статьи.

        Args:
            title: заголовок статьи
            content: полный текст (берём первый абзац)
            tags: теги статьи

        Returns:
            ImagePromptResult с промптом на английском
        """
        # Берём первый абзац (до 500 символов)
        first_paragraph = self._extract_first_paragraph(content, max_chars=500)
        tags_str = ", ".join(tags[:5]) if tags else "technology"

        logger.info(f"[ImagePrompt] Title: {title[:50]}...")
        logger.info(f"[ImagePrompt] Tags: {tags_str}")

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
                temperature=0.7,  # Немного креативности
                min_response_length=20,
            )

            cleaned = self._clean_prompt(result)

            if len(cleaned) < 20:
                logger.warning(f"[ImagePrompt] Слишком короткий: {len(cleaned)}")
                return self._fallback_prompt(title, tags)

            logger.info(f"[ImagePrompt] OK: {len(cleaned)} chars")
            logger.info(f"[ImagePrompt] Prompt: {cleaned[:100]}...")

            return ImagePromptResult(
                prompt=cleaned,
                style="digital art",
                success=True,
            )

        except Exception as e:
            logger.warning(f"[ImagePrompt] Ошибка: {e}")
            return self._fallback_prompt(title, tags)

    def _extract_first_paragraph(self, content: str, max_chars: int = 500) -> str:
        """Извлечь первый абзац из текста."""
        if not content:
            return ""

        # Убираем заголовки markdown
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

        result = text.strip()

        # Убираем кавычки
        result = result.strip('"\'')

        # Убираем префиксы
        prefixes = [
            r'^IMAGE PROMPT:?\s*',
            r'^Prompt:?\s*',
            r'^Here is.*?:\s*',
            r'^A\s+',
        ]
        for p in prefixes:
            result = re.sub(p, '', result, flags=re.IGNORECASE)

        # Убираем переносы строк — промпт должен быть одной строкой
        result = re.sub(r'\n+', ' ', result)
        result = re.sub(r'\s+', ' ', result)

        return result.strip()

    def _fallback_prompt(self, title: str, tags: Optional[List[str]] = None) -> ImagePromptResult:
        """Fallback промпт если LLM не справился."""
        tag_words = " ".join(tags[:3]) if tags else "technology code"

        prompt = (
            f"Modern digital illustration for tech blog, "
            f"abstract geometric composition representing {tag_words}, "
            f"clean minimalist design, blue and purple gradient, "
            f"floating 3D shapes, glowing connections, "
            f"professional header image, no text, no faces"
        )

        logger.info(f"[ImagePrompt] Fallback: {prompt[:80]}...")
        return ImagePromptResult(prompt=prompt, style="digital art", success=False)

    def process(self, title: str, content: str, tags: List[str] = None) -> ImagePromptResult:
        """Основной метод для оркестратора."""
        return self.generate_prompt(title, content, tags)