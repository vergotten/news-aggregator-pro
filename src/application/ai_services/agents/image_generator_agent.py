# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/image_generator_agent.py
# =============================================================================
"""
Агент генерации изображений v1.0

Шаг 10 в оркестраторе: берёт промпт из Step 9 (ImagePromptAgent)
и генерирует обложку через Pollinations.ai (бесплатный API).

Pollinations.ai:
- Бесплатный, без API ключа
- FLUX модель
- Просто GET запрос с промптом в URL
- Возвращает PNG изображение

Результат: URL картинки → telegram_cover_image
"""

import logging
import os
import time
import hashlib
from typing import Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

# Pollinations.ai — бесплатный image gen API
POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"
DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 630
REQUEST_TIMEOUT = 60  # секунд


class ImageGeneratorResult:
    """Результат генерации изображения."""
    def __init__(self, url: str = "", local_path: str = "", success: bool = False, error: str = ""):
        self.url = url
        self.local_path = local_path
        self.success = success
        self.error = error


class ImageGeneratorAgent:
    """
    Агент генерации обложек v1.0

    Не наследует BaseAgent — не использует LLM.
    Просто HTTP запрос к Pollinations.ai.
    """

    agent_name = "image_generator"

    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        self.width = width
        self.height = height
        logger.info(f"[INIT] ImageGeneratorAgent v1.0 (Pollinations.ai, {width}x{height})")

    def generate_image(
        self,
        prompt: str,
        article_id: str = "",
        save_dir: str = "",
    ) -> ImageGeneratorResult:
        """
        Генерировать изображение по промпту.

        Args:
            prompt: промпт на английском (из ImagePromptAgent)
            article_id: ID статьи (для имени файла)
            save_dir: директория для сохранения (опционально)

        Returns:
            ImageGeneratorResult с URL и/или локальным путём
        """
        if not prompt:
            return ImageGeneratorResult(error="Пустой промпт", success=False)

        logger.info(f"[ImageGen] Prompt: {prompt[:80]}...")

        # Формируем URL
        encoded_prompt = quote(prompt)
        url = POLLINATIONS_URL.format(prompt=encoded_prompt)
        url += f"?width={self.width}&height={self.height}&nologo=true"

        logger.info(f"[ImageGen] Запрос к Pollinations.ai...")

        try:
            start = time.time()
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            elapsed = time.time() - start

            if resp.status_code != 200:
                error = f"HTTP {resp.status_code}"
                logger.error(f"[ImageGen] Ошибка: {error}")
                return ImageGeneratorResult(error=error, success=False)

            # Проверяем что это изображение
            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type:
                error = f"Не изображение: {content_type}"
                logger.error(f"[ImageGen] {error}")
                return ImageGeneratorResult(error=error, success=False)

            image_data = resp.content
            image_size = len(image_data)

            if image_size < 1000:
                error = f"Слишком маленькое: {image_size} bytes"
                logger.error(f"[ImageGen] {error}")
                return ImageGeneratorResult(error=error, success=False)

            logger.info(f"[ImageGen] OK: {image_size/1024:.0f} KB, {elapsed:.1f}s")

            # Загружаем на Telegraph для получения постоянного URL
            telegraph_url = self._upload_to_telegraph(image_data)

            if telegraph_url:
                logger.info(f"[ImageGen] Telegraph URL: {telegraph_url}")
                result = ImageGeneratorResult(
                    url=telegraph_url,
                    success=True,
                )
            else:
                # Fallback: используем Pollinations URL напрямую
                # (работает но может быть нестабильным)
                direct_url = url
                logger.warning("[ImageGen] Telegraph upload не удался, используем Pollinations URL")
                result = ImageGeneratorResult(
                    url=direct_url,
                    success=True,
                )

            # Опционально сохраняем локально
            if save_dir:
                filename = f"{article_id[:8] or 'cover'}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}.png"
                local_path = os.path.join(save_dir, filename)
                with open(local_path, "wb") as f:
                    f.write(image_data)
                result.local_path = local_path
                logger.info(f"[ImageGen] Saved: {local_path}")

            return result

        except requests.Timeout:
            error = f"Timeout ({REQUEST_TIMEOUT}s)"
            logger.error(f"[ImageGen] {error}")
            return ImageGeneratorResult(error=error, success=False)

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.error(f"[ImageGen] {error}")
            return ImageGeneratorResult(error=error, success=False)

    def _upload_to_telegraph(self, image_data: bytes) -> Optional[str]:
        """
        Загрузить изображение на Telegraph для постоянного URL.

        Telegraph принимает файлы через multipart upload.
        Возвращает URL вида https://telegra.ph/file/xxx.png
        """
        try:
            resp = requests.post(
                "https://telegra.ph/upload",
                files={"file": ("cover.png", image_data, "image/png")},
                timeout=30,
            )

            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and len(result) > 0:
                    src = result[0].get("src", "")
                    if src:
                        return f"https://telegra.ph{src}"

            logger.warning(f"[ImageGen] Telegraph upload: HTTP {resp.status_code}")
            return None

        except Exception as e:
            logger.warning(f"[ImageGen] Telegraph upload error: {e}")
            return None
