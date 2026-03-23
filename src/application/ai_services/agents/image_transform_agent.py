# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/image_transform_agent.py
# =============================================================================
"""
Агент трансформации изображений v1.0

Берёт изображения из статьи и слегка трансформирует каждое,
чтобы они не были идентичны оригиналу (для Дзена и антиплагиата).

Трансформации (все лёгкие, не портят качество):
- Зум/crop 5-15% от центра
- Rotation ±1-3°
- Яркость ±5-10%
- Контраст ±5-10%
- Цветовой сдвиг (hue) ±5-10°
- Лёгкий gaussian noise
- Resize up-down (меняет пиксели)
- JPEG recompression (другой quality)
- Лёгкий vignette (затемнение краёв)
- Гамма-коррекция

Не использует LLM, не требует GPU.
Зависимости: Pillow, numpy, requests

Результат: список URL трансформированных изображений (загруженных на Telegraph).
"""

import io
import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# Параметры трансформаций
CROP_RANGE = (0.05, 0.12)       # 5-12% зум
ROTATION_RANGE = (-2.5, 2.5)    # ±2.5°
BRIGHTNESS_RANGE = (0.92, 1.08)  # ±8%
CONTRAST_RANGE = (0.93, 1.07)    # ±7%
SATURATION_RANGE = (0.93, 1.07)  # ±7%
NOISE_STRENGTH = 3               # sigma для gaussian noise
GAMMA_RANGE = (0.92, 1.08)       # гамма-коррекция
JPEG_QUALITY_RANGE = (85, 93)    # качество JPEG при сохранении
VIGNETTE_STRENGTH = 0.15         # сила затемнения краёв

REQUEST_TIMEOUT = 30


class ImageTransformResult:
    """Результат трансформации изображений."""
    def __init__(self):
        self.original_urls: List[str] = []
        self.transformed_urls: List[str] = []
        self.success_count: int = 0
        self.error_count: int = 0
        self.errors: List[str] = []


class ImageTransformAgent:
    """
    Агент трансформации изображений v1.0

    Не наследует BaseAgent — не использует LLM.
    Чистый PIL/numpy для обработки изображений.
    """

    agent_name = "image_transform"

    def __init__(self, upload_to_telegraph: bool = True):
        self.upload_to_telegraph = upload_to_telegraph
        logger.info("[INIT] ImageTransformAgent v1.0")

    def transform_images(self, image_urls: List[str]) -> ImageTransformResult:
        """
        Трансформировать список изображений.

        Args:
            image_urls: список URL оригинальных изображений

        Returns:
            ImageTransformResult с новыми URL
        """
        result = ImageTransformResult()
        result.original_urls = list(image_urls)

        if not image_urls:
            return result

        logger.info(f"[ImageTransform] Обработка {len(image_urls)} изображений")

        for i, url in enumerate(image_urls, 1):
            try:
                logger.info(f"[ImageTransform] [{i}/{len(image_urls)}] {url[:60]}...")

                # Скачиваем
                img = self._download_image(url)
                if img is None:
                    result.error_count += 1
                    result.errors.append(f"Не удалось скачать: {url[:60]}")
                    result.transformed_urls.append(url)  # оставляем оригинал
                    continue

                # Трансформируем
                transformed = self._apply_transforms(img)

                # Загружаем обратно
                if self.upload_to_telegraph:
                    new_url = self._upload_to_telegraph(transformed)
                    if new_url:
                        result.transformed_urls.append(new_url)
                        result.success_count += 1
                        logger.info(f"[ImageTransform] OK: {new_url}")
                        continue

                # Fallback: оставляем оригинал
                result.transformed_urls.append(url)
                result.error_count += 1

            except Exception as e:
                logger.warning(f"[ImageTransform] Ошибка [{i}]: {e}")
                result.transformed_urls.append(url)
                result.error_count += 1
                result.errors.append(str(e))

        logger.info(
            f"[ImageTransform] Готово: {result.success_count} OK, "
            f"{result.error_count} ошибок из {len(image_urls)}"
        )
        return result

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Скачать изображение по URL."""
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning(f"[ImageTransform] HTTP {resp.status_code}: {url[:60]}")
                return None

            img = Image.open(io.BytesIO(resp.content))
            # Конвертируем в RGB если нужно
            if img.mode in ('RGBA', 'P', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA' or img.mode == 'LA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            return img

        except Exception as e:
            logger.warning(f"[ImageTransform] Download error: {e}")
            return None

    def _apply_transforms(self, img: Image.Image) -> Image.Image:
        """Применить набор лёгких трансформаций."""
        original_size = img.size

        # 1. Crop/Zoom (5-12%)
        img = self._apply_crop(img)

        # 2. Rotation (±2.5°)
        img = self._apply_rotation(img)

        # 3. Brightness (±8%)
        img = self._apply_brightness(img)

        # 4. Contrast (±7%)
        img = self._apply_contrast(img)

        # 5. Saturation/Color (±7%)
        img = self._apply_saturation(img)

        # 6. Gaussian noise
        img = self._apply_noise(img)

        # 7. Gamma correction
        img = self._apply_gamma(img)

        # 8. Vignette (затемнение краёв)
        img = self._apply_vignette(img)

        # 9. Resize up-down (меняет пиксели через интерполяцию)
        img = self._apply_resize_trick(img)

        # Вернуть к оригинальному размеру
        if img.size != original_size:
            img = img.resize(original_size, Image.LANCZOS)

        return img

    def _apply_crop(self, img: Image.Image) -> Image.Image:
        """Зум: обрезать края на 5-12%."""
        w, h = img.size
        crop_pct = random.uniform(*CROP_RANGE)
        left = int(w * crop_pct / 2)
        top = int(h * crop_pct / 2)
        right = w - left
        bottom = h - top
        cropped = img.crop((left, top, right, bottom))
        return cropped.resize((w, h), Image.LANCZOS)

    def _apply_rotation(self, img: Image.Image) -> Image.Image:
        """Лёгкий поворот ±2.5°."""
        angle = random.uniform(*ROTATION_RANGE)
        return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

    def _apply_brightness(self, img: Image.Image) -> Image.Image:
        """Яркость ±8%."""
        factor = random.uniform(*BRIGHTNESS_RANGE)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    def _apply_contrast(self, img: Image.Image) -> Image.Image:
        """Контраст ±7%."""
        factor = random.uniform(*CONTRAST_RANGE)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def _apply_saturation(self, img: Image.Image) -> Image.Image:
        """Насыщенность ±7%."""
        factor = random.uniform(*SATURATION_RANGE)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)

    def _apply_noise(self, img: Image.Image) -> Image.Image:
        """Лёгкий gaussian noise."""
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, NOISE_STRENGTH, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _apply_gamma(self, img: Image.Image) -> Image.Image:
        """Гамма-коррекция."""
        gamma = random.uniform(*GAMMA_RANGE)
        inv_gamma = 1.0 / gamma
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.power(arr, inv_gamma)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _apply_vignette(self, img: Image.Image) -> Image.Image:
        """Затемнение краёв (vignette)."""
        w, h = img.size
        arr = np.array(img, dtype=np.float32)

        # Создаём маску затемнения
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        # Нормализованное расстояние от центра (0 в центре, 1 на углах)
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        dist = dist / max_dist

        # Применяем затемнение только к краям
        mask = 1.0 - VIGNETTE_STRENGTH * (dist ** 2)
        mask = mask[:, :, np.newaxis]  # добавляем ось для RGB

        arr = arr * mask
        arr = arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _apply_resize_trick(self, img: Image.Image) -> Image.Image:
        """Resize up потом down — меняет пиксели через интерполяцию."""
        w, h = img.size
        scale = random.uniform(1.08, 1.15)
        big = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return big.resize((w, h), Image.LANCZOS)

    def _upload_to_telegraph(self, img: Image.Image) -> Optional[str]:
        """Загрузить трансформированное изображение на Telegraph."""
        try:
            buf = io.BytesIO()
            quality = random.randint(*JPEG_QUALITY_RANGE)
            img.save(buf, format='JPEG', quality=quality, optimize=True)
            buf.seek(0)

            resp = requests.post(
                "https://telegra.ph/upload",
                files={"file": ("image.jpg", buf, "image/jpeg")},
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and len(result) > 0:
                    src = result[0].get("src", "")
                    if src:
                        return f"https://telegra.ph{src}"

            logger.warning(f"[ImageTransform] Telegraph upload: HTTP {resp.status_code}")
            return None

        except Exception as e:
            logger.warning(f"[ImageTransform] Telegraph upload error: {e}")
            return None