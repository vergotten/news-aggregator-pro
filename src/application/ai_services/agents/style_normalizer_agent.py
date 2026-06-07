# -*- coding: utf-8 -*-
"""
Агент нормализации стиля v9.4 - PRODUCTION FIX

Ключевые изменения от v9.3:
1. МЕНЬШИЕ чанки (8000 символов) - модель лучше справляется
2. РЕАЛИСТИЧНЫЙ min_response (30% вместо 50%)
3. Улучшенный промпт с явными инструкциями
4. Fallback на minimal_cleanup если модель не справляется
5. Увеличен timeout в запросах
"""

import logging
import re
from typing import Optional, List
from pydantic import BaseModel, Field

from src.application.ai_services.agents.base_agent import BaseAgent, TaskType, ContextLimitError

logger = logging.getLogger(__name__)

# =============================================================================
# КРИТИЧЕСКИЕ НАСТРОЙКИ ДЛЯ OLLAMA
# =============================================================================
# qwen2.5:14b часто останавливается рано на длинных текстах
# Решение: меньшие чанки + менее строгие требования к длине ответа

MAX_CONTENT_LENGTH = 8000
MAX_CHUNK_SIZE = 6000
CHUNK_OVERLAP = 200

# Минимальный ответ = 30% от входа (было 50%)
MIN_RESPONSE_RATIO = 0.3

# Если ответ короче этого - используем fallback
ABSOLUTE_MIN_RESPONSE = 200


class NormalizationResult(BaseModel):
    """Результат нормализации."""
    normalized_text: str = Field(description="Нормализованный текст")
    changes_made: list[str] = Field(default_factory=list)
    personal_pronouns_removed: int = Field(default=0)
    length_ratio: float = Field(default=1.0)
    chunks_processed: int = Field(default=1)
    used_fallback: bool = Field(default=False)


class StyleNormalizationResult(BaseModel):
    """Результат для orchestrator."""
    normalized_text: str = Field(description="Нормализованный текст")
    original_issues: List[str] = Field(default_factory=list)
    improvements_made: List[str] = Field(default_factory=list)
    processing_time: Optional[float] = Field(default=None)


class StyleNormalizerAgent(BaseAgent):
    """
    Агент нормализации стиля

    Оптимизирован для работы с Ollama/qwen2.5:14b:
    - Маленькие чанки (6-8K символов)
    - Реалистичные ожидания по длине ответа
    - Агрессивный fallback на minimal_cleanup
    """

    agent_name = "style_normalizer"
    task_type = TaskType.HEAVY
    MIN_RESPONSE_LENGTH = ABSOLUTE_MIN_RESPONSE

    EXCLUDED_MODELS = [
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "meta-llama/llama-3.2-3b",
        "meta-llama/llama-3.1-8b",
    ]

    # Короткий и чёткий системный промпт
    SYSTEM_PROMPT = """Ты редактор. Преобразуй текст в безличный стиль.
Удали: приветствия, прощания, "я/мы/мой".
Сохрани: всю техническую информацию.
Выведи ТОЛЬКО результат, без комментариев."""

    def __init__(self, llm_provider=None, config=None, **kwargs):
        super().__init__(
            llm_provider=llm_provider,
            config=config,
            max_retries=2,  # Меньше попыток - быстрее fallback
            retry_delay=2.0
        )
        logger.info(f"[INIT] StyleNormalizerAgent v9.4 (PRODUCTION)")

    def normalize_full_text(self, content: str, url: str = "") -> str:
        """Нормализовать текст -> plain text."""
        result = self.normalize_with_details(content, url)
        return result.normalized_text

    def normalize_with_details(self, content: str, url: str = "") -> NormalizationResult:
        """Нормализовать с метриками."""
        original_length = len(content)

        if original_length < 100:
            return NormalizationResult(
                normalized_text=content,
                changes_made=["Текст слишком короткий"],
                length_ratio=1.0
            )

        # Всегда используем chunking для текстов > MAX_CONTENT_LENGTH
        if original_length > MAX_CONTENT_LENGTH:
            logger.info(
                f"[StyleNormalizer] Long text ({original_length} chars), chunking..."
            )
            return self._process_with_chunks(content, original_length, url)
        else:
            return self._process_single(content, original_length, url)

    def process(self, text: str, url: str = "") -> StyleNormalizationResult:
        """Основной метод для orchestrator."""
        import time
        start_time = time.time()

        try:
            result = self.normalize_with_details(text, url)
            processing_time = time.time() - start_time

            improvements = []
            if result.personal_pronouns_removed > 0:
                improvements.append(f"Удалено {result.personal_pronouns_removed} местоимений")
            if result.chunks_processed > 1:
                improvements.append(f"Обработано {result.chunks_processed} частей")
            if result.used_fallback:
                improvements.append("Использована базовая очистка")
            else:
                improvements.append("Текст нормализован")

            return StyleNormalizationResult(
                normalized_text=result.normalized_text,
                original_issues=[],
                improvements_made=improvements,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[StyleNormalizer] Error: {e}")

            # При любой ошибке - fallback
            cleaned = self._minimal_cleanup(text)
            return StyleNormalizationResult(
                normalized_text=cleaned,
                original_issues=[f"Ошибка: {str(e)[:100]}"],
                improvements_made=["Базовая очистка"],
                processing_time=processing_time
            )

    def _process_single(self, content: str, original_length: int, url: str = "") -> NormalizationResult:
        """Обработка одного текста."""

        # Для Ollama: max_tokens = примерно столько же сколько на входе
        # 1 токен ≈ 2-3 символа для русского
        max_tokens = max(2048, min(8192, original_length // 2 + 1000))

        # Реалистичный минимум: 30% от оригинала
        min_response = max(ABSOLUTE_MIN_RESPONSE, int(original_length * MIN_RESPONSE_RATIO))

        prompt = self._build_prompt(content)

        logger.info(
            f"[StyleNormalizer] Processing {original_length} chars, "
            f"max_tokens={max_tokens}, min={min_response}"
        )

        try:
            # Используем BaseAgent.generate() с retry
            normalized = self.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=0.2,  # Очень низкая для консистентности
                min_response_length=min_response
            )

            # Очистка
            cleaned = self._clean_result(normalized)

            # Проверка качества
            if len(cleaned) < min_response:
                logger.warning(
                    f"[StyleNormalizer] Short result ({len(cleaned)}<{min_response}), fallback"
                )
                cleaned = self._minimal_cleanup(content)
                return NormalizationResult(
                    normalized_text=cleaned,
                    changes_made=["Fallback: short response"],
                    length_ratio=len(cleaned) / original_length,
                    used_fallback=True
                )

            pronouns = self._count_pronouns_diff(content, cleaned)

            logger.info(
                f"[StyleNormalizer] OK: {original_length} -> {len(cleaned)} "
                f"(ratio={len(cleaned) / original_length:.1%})"
            )

            return NormalizationResult(
                normalized_text=cleaned,
                changes_made=["Нормализация"],
                personal_pronouns_removed=pronouns,
                length_ratio=len(cleaned) / original_length
            )

        except Exception as e:
            logger.warning(f"[StyleNormalizer] LLM failed: {e}, using fallback")
            cleaned = self._minimal_cleanup(content)
            return NormalizationResult(
                normalized_text=cleaned,
                changes_made=[f"Fallback: {type(e).__name__}"],
                length_ratio=len(cleaned) / original_length,
                used_fallback=True
            )

    def _process_with_chunks(self, content: str, original_length: int, url: str = "") -> NormalizationResult:
        """Обработка длинного текста по частям."""
        chunks = self._split_into_chunks(content)
        logger.info(f"[StyleNormalizer] Split into {len(chunks)} chunks")

        normalized_chunks = []
        total_pronouns = 0
        any_fallback = False

        for i, chunk in enumerate(chunks):
            chunk_len = len(chunk)
            logger.info(f"[StyleNormalizer] Chunk {i + 1}/{len(chunks)} ({chunk_len} chars)")

            try:
                result = self._process_single(chunk, chunk_len, "")
                normalized_chunks.append(result.normalized_text)
                total_pronouns += result.personal_pronouns_removed
                if result.used_fallback:
                    any_fallback = True
            except Exception as e:
                logger.warning(f"[StyleNormalizer] Chunk {i + 1} error: {e}")
                normalized_chunks.append(self._minimal_cleanup(chunk))
                any_fallback = True

        # Собираем результат
        normalized = "\n\n".join(normalized_chunks)
        ratio = len(normalized) / original_length if original_length > 0 else 1.0

        logger.info(
            f"[StyleNormalizer] Done: {original_length} -> {len(normalized)} "
            f"({len(chunks)} chunks, fallback={any_fallback})"
        )

        return NormalizationResult(
            normalized_text=normalized,
            changes_made=[f"{len(chunks)} chunks"],
            personal_pronouns_removed=total_pronouns,
            length_ratio=ratio,
            chunks_processed=len(chunks),
            used_fallback=any_fallback
        )

    def _build_prompt(self, content: str) -> str:
        """Промпт для нормализации - короткий и чёткий."""
        return f"""Преобразуй текст в редакционный стиль:

1. Удали "Привет", "Всем привет", "Спасибо за внимание"
2. Замени "я сделал" → "было сделано"  
3. Замени "мы решили" → "было решено"
4. Сохрани всю техническую информацию

ТЕКСТ:
{content}

РЕЗУЛЬТАТ:"""

    def _split_into_chunks(self, content: str) -> List[str]:
        """Разбить на чанки по абзацам."""
        paragraphs = content.split('\n\n')
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Если добавление параграфа превысит лимит
            if len(current) + len(para) + 2 > MAX_CHUNK_SIZE:
                if current:
                    chunks.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para

        if current:
            chunks.append(current.strip())

        # Если получился один большой чанк - разбиваем по предложениям
        if len(chunks) == 1 and len(chunks[0]) > MAX_CHUNK_SIZE:
            chunks = self._split_by_sentences(chunks[0])

        # Если чанков нет - вернуть весь текст как один
        if not chunks:
            chunks = [content]

        return chunks

    def _split_by_sentences(self, content: str) -> List[str]:
        """Разбить по предложениям."""
        # Разбиваем по точке/восклицательному/вопросительному + пробел
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) + 1 > MAX_CHUNK_SIZE:
                if current:
                    chunks.append(current.strip())
                current = sent
            else:
                current = current + " " + sent if current else sent

        if current:
            chunks.append(current.strip())

        return chunks if chunks else [content]

    def _clean_result(self, text: str) -> str:
        """Очистка от артефактов LLM."""
        if not text:
            return ""

        result = text.strip()

        # Убираем markdown
        result = re.sub(r'^```[\w]*\n?', '', result)
        result = re.sub(r'\n?```$', '', result)

        # Убираем префиксы LLM
        prefixes = [
            r'^(?:Вот\s+)?(?:преобразованный|нормализованный|результат)[:\s]*\n*',
            r'^РЕЗУЛЬТАТ[:\s]*\n*',
            r'^Редакционный\s+(?:вариант|текст)[:\s]*\n*',
        ]
        for p in prefixes:
            result = re.sub(p, '', result, flags=re.IGNORECASE)

        # Убираем markdown разметку
        result = re.sub(r'\*\*([^*]+)\*\*', r'\1', result)
        result = re.sub(r'\*([^*]+)\*', r'\1', result)
        result = re.sub(r'__([^_]+)__', r'\1', result)
        result = re.sub(r'_([^_]+)_', r'\1', result)

        return result.strip()

    def _minimal_cleanup(self, content: str) -> str:
        """Минимальная очистка без LLM - всегда работает."""
        text = content

        # Приветствия в начале
        greetings = [
            r'^\s*Привет[,!]?\s*(?:Хабр|всем)?[,!.]?\s*\n*',
            r'^\s*Всем\s+привет[,!.]?\s*\n*',
            r'^\s*Добрый\s+день[,!.]?\s*\n*',
            r'^\s*Здравствуйте[,!.]?\s*\n*',
            r'^\s*Приветствую[,!.]?\s*\n*',
        ]
        for pattern in greetings:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Прощания в конце
        farewells = [
            r'\n*Спасибо\s+за\s+(?:внимание|прочтение)[,!.]*\s*$',
            r'\n*Подписывайтесь[^.]*[,!.]*\s*$',
            r'\n*Пишите\s+в\s+комментариях[^.]*[,!.]*\s*$',
            r'\n*Буду\s+рад\s+(?:вашим\s+)?комментариям[^.]*[,!.]*\s*$',
            r'\n*Жду\s+ваших?\s+(?:комментари|отзыв)[^.]*[,!.]*\s*$',
        ]
        for pattern in farewells:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text.strip()

    def _count_pronouns_diff(self, original: str, normalized: str) -> int:
        """Подсчёт удалённых местоимений."""
        patterns = [
            r'\bя\b', r'\bмы\b', r'\bмой\b', r'\bмоя\b', r'\bмоё\b', r'\bмои\b',
            r'\bнаш\b', r'\bнаша\b', r'\bнаше\b', r'\bнаши\b',
            r'\bмне\b', r'\bнам\b', r'\bменя\b', r'\bнас\b',
        ]

        def count(text: str) -> int:
            total = 0
            text_lower = text.lower()
            for p in patterns:
                total += len(re.findall(p, text_lower))
            return total

        return max(0, count(original) - count(normalized))