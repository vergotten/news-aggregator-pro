# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/image_transform_agent.py
# =============================================================================
"""
Агент трансформации изображений v2.0

Мягкие трансформации для уникальности картинок (Дзен антиплагиат).
Загрузка результатов в Supabase Storage.

Трансформации (мягкие, не портят текст/графики):
- Brightness ±5%
- Contrast ±5%
- Saturation ±5%
- Gaussian noise (sigma=2-3)
- Gamma correction ±5%
- JPEG recompress (quality 85-93)
- Resize up-down (интерполяция меняет пиксели)

Включение/выключение: ENABLE_IMAGE_TRANSFORM=true/false в .env

Зависимости: Pillow, numpy, requests
"""

import io
import os
import logging
import random
import hashlib
import time
from typing import List, Optional

import numpy as np
import requests
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

BRIGHTNESS_RANGE = (0.95, 1.05)
CONTRAST_RANGE = (0.95, 1.05)
SATURATION_RANGE = (0.95, 1.05)
NOISE_STRENGTH = 2.5
GAMMA_RANGE = (0.95, 1.05)
JPEG_QUALITY_RANGE = (85, 93)
REQUEST_TIMEOUT = 30

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://vhsnoprsdxvcztdickfd.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_BUCKET = "images"


def is_enabled() -> bool:
    val = os.getenv("ENABLE_IMAGE_TRANSFORM", "true").lower()
    return val in ("true", "1", "yes", "on")


class ImageTransformResult:
    def __init__(self):
        self.original_urls: List[str] = []
        self.transformed_urls: List[str] = []
        self.success_count: int = 0
        self.error_count: int = 0
        self.skipped: bool = False
        self.errors: List[str] = []


class ImageTransformAgent:
    agent_name = "image_transform"

    def __init__(self):
        self.supabase_url = SUPABASE_URL
        self.supabase_key = SUPABASE_ANON_KEY or self._load_key_from_env()
        self.bucket = SUPABASE_BUCKET
        self.enabled = is_enabled()
        logger.info(f"[INIT] ImageTransformAgent v2.0 (enabled={self.enabled}, storage=Supabase)")

    def _load_key_from_env(self) -> str:
        for env_path in [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".env"),
            os.path.join(os.getcwd(), ".env"),
        ]:
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("SUPABASE_ANON_KEY="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")
        return ""

    def transform_images(self, image_urls: List[str]) -> ImageTransformResult:
        result = ImageTransformResult()
        result.original_urls = list(image_urls)

        if not image_urls:
            return result

        if not self.enabled:
            logger.info("[ImageTransform] ОТКЛЮЧЕН (ENABLE_IMAGE_TRANSFORM=false)")
            result.transformed_urls = list(image_urls)
            result.skipped = True
            return result

        if not self.supabase_key:
            logger.warning("[ImageTransform] SUPABASE_ANON_KEY не задан, пропуск")
            result.transformed_urls = list(image_urls)
            result.skipped = True
            return result

        logger.info(f"[ImageTransform] Обработка {len(image_urls)} изображений")

        for i, url in enumerate(image_urls, 1):
            try:
                logger.info(f"[ImageTransform] [{i}/{len(image_urls)}] {url[:60]}...")

                img = self._download_image(url)
                if img is None:
                    result.error_count += 1
                    result.errors.append(f"Download failed: {url[:60]}")
                    result.transformed_urls.append(url)
                    continue

                transformed = self._apply_soft_transforms(img)

                filename = self._generate_filename(url, i)
                new_url = self._upload_to_supabase(transformed, filename)

                if new_url:
                    result.transformed_urls.append(new_url)
                    result.success_count += 1
                    logger.info(f"[ImageTransform] OK → {new_url[:60]}...")
                else:
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
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning(f"[ImageTransform] HTTP {resp.status_code}: {url[:60]}")
                return None

            img = Image.open(io.BytesIO(resp.content))

            if img.mode in ('RGBA', 'P', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode in ('RGBA', 'LA'):
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

    def _apply_soft_transforms(self, img: Image.Image) -> Image.Image:
        # Brightness ±5%
        img = ImageEnhance.Brightness(img).enhance(random.uniform(*BRIGHTNESS_RANGE))

        # Contrast ±5%
        img = ImageEnhance.Contrast(img).enhance(random.uniform(*CONTRAST_RANGE))

        # Saturation ±5%
        img = ImageEnhance.Color(img).enhance(random.uniform(*SATURATION_RANGE))

        # Gaussian noise
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, NOISE_STRENGTH, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        # Gamma correction ±5%
        gamma = random.uniform(*GAMMA_RANGE)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.power(arr, 1.0 / gamma)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        # Resize up-down
        w, h = img.size
        scale = random.uniform(1.08, 1.12)
        big = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img = big.resize((w, h), Image.LANCZOS)

        return img

    def _generate_filename(self, original_url: str, index: int) -> str:
        url_hash = hashlib.md5(original_url.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        return f"img_{timestamp}_{url_hash}_{index}.jpg"

    def _upload_to_supabase(self, img: Image.Image, filename: str) -> Optional[str]:
        try:
            buf = io.BytesIO()
            quality = random.randint(*JPEG_QUALITY_RANGE)
            img.save(buf, format='JPEG', quality=quality, optimize=True)
            buf.seek(0)
            image_data = buf.read()

            headers = {
                'Authorization': f'Bearer {self.supabase_key}',
                'apikey': self.supabase_key,
                'Content-Type': 'image/jpeg',
            }

            upload_url = f"{self.supabase_url}/storage/v1/object/{self.bucket}/{filename}"
            resp = requests.post(upload_url, headers=headers, data=image_data, timeout=REQUEST_TIMEOUT)

            if resp.status_code in (200, 201):
                return f"{self.supabase_url}/storage/v1/object/public/{self.bucket}/{filename}"
            else:
                logger.warning(f"[ImageTransform] Supabase upload {resp.status_code}: {resp.text[:100]}")
                return None

        except Exception as e:
            logger.warning(f"[ImageTransform] Supabase upload error: {e}")
            return None