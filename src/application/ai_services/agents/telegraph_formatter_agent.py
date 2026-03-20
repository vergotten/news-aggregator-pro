# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/telegraph_formatter_agent.py
# =============================================================================
"""
Агент форматирования для Telegraph v2.0

Шаг 8 в оркестраторе: берёт editorial_rewritten (plain text)
и создаёт размеченный markdown для Telegraph.

Работает по чанкам (как StyleNormalizer):
1. Рассчитывает коэффициент сжатия исходя из общей длины текста
2. Каждый чанк: сжать + разметить markdown за один вызов LLM
3. Склеить + верифицировать итоговый размер

Лимит Telegraph: ~34K символов → с учётом JSON overhead цель <= 22K

Изменения v2.0:
- Chunking для больших текстов
- Адаптивный коэффициент сжатия
- Сжатие + разметка в одном вызове LLM
- Верификация итогового размера
"""

import logging
import re
from typing import Optional, List
from pydantic import BaseModel, Field

from src.application.ai_services.agents.base_agent import BaseAgent, TaskType

logger = logging.getLogger(__name__)

MAX_TELEGRAPH_CHARS = 22000
MAX_CHUNK_SIZE = 6000
CHUNK_OVERLAP = 100


class TelegraphFormatResult(BaseModel):
    formatted_text: str = Field(description="Текст с markdown разметкой")
    original_length: int = Field(default=0)
    final_length: int = Field(default=0)
    chunks_processed: int = Field(default=0)
    compression_ratio: float = Field(default=1.0)
    was_compressed: bool = Field(default=False)
    was_formatted: bool = Field(default=False)


class TelegraphFormatterAgent(BaseAgent):
    agent_name = "telegraph_formatter"
    task_type = TaskType.HEAVY
    MIN_RESPONSE_LENGTH = 200

    SYSTEM_PROMPT = """Ты — технический редактор. Сожми и отформатируй текст для публикации.
Выведи ТОЛЬКО результат, без комментариев."""

    COMPRESS_AND_FORMAT_PROMPT = """Сожми текст до {target_chars} символов И добавь markdown-разметку.

ПРАВИЛА СЖАТИЯ:
- Убери повторяющиеся объяснения и многословие
- Убери вводные фразы типа "давайте рассмотрим", "стоит отметить"
- Сохрани ВСЕ блоки кода полностью
- Сохрани технические термины и ключевые факты

ПРАВИЛА РАЗМЕТКИ:
- Блоки кода оберни в ```язык ... ```
- Подзаголовки пометь через ## 
- Списки через -
- Inline код через `backticks`
- НЕ добавляй заголовок статьи

ТЕКСТ ({current_chars} символов, нужно {target_chars}):
{content}

РЕЗУЛЬТАТ:"""

    FORMAT_ONLY_PROMPT = """Добавь markdown-разметку в текст.

ПРАВИЛА:
- Блоки кода оберни в ```язык ... ```
- Подзаголовки пометь через ##
- Списки через -
- Inline код через `backticks`
- НЕ меняй содержание, только добавляй разметку
- НЕ добавляй заголовок статьи

ТЕКСТ:
{content}

РАЗМЕЧЕННЫЙ ТЕКСТ:"""

    def __init__(self, llm_provider=None, config=None, **kwargs):
        super().__init__(
            llm_provider=llm_provider,
            config=config,
            max_retries=2,
            retry_delay=3.0,
        )
        logger.info("[INIT] TelegraphFormatterAgent v2.0")

    def format_for_telegraph(self, content: str) -> TelegraphFormatResult:
        if not content or len(content) < 100:
            return TelegraphFormatResult(
                formatted_text=content or "",
                original_length=len(content or ""),
                final_length=len(content or ""),
            )

        original_length = len(content)
        logger.info(f"[TelegraphFormatter] Вход: {original_length} chars")

        # Рассчитываем нужно ли сжатие
        needs_compression = original_length > MAX_TELEGRAPH_CHARS
        if needs_compression:
            target_ratio = MAX_TELEGRAPH_CHARS / original_length
            target_ratio = max(0.4, target_ratio)
            logger.info(
                f"[TelegraphFormatter] Сжатие: {original_length} -> ~{MAX_TELEGRAPH_CHARS} "
                f"(ratio={target_ratio:.0%})"
            )
        else:
            target_ratio = 1.0
            logger.info("[TelegraphFormatter] Только разметка (сжатие не нужно)")

        # Разбиваем на чанки
        chunks = self._split_into_chunks(content)
        logger.info(f"[TelegraphFormatter] {len(chunks)} чанков")

        # Обрабатываем каждый чанк
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            chunk_len = len(chunk)
            target_chunk_len = int(chunk_len * target_ratio)

            logger.info(
                f"[TelegraphFormatter] Чанк {i}/{len(chunks)} "
                f"({chunk_len} -> ~{target_chunk_len} chars)"
            )

            result = self._process_chunk(
                chunk=chunk,
                chunk_len=chunk_len,
                target_len=target_chunk_len,
                needs_compression=needs_compression,
            )
            formatted_chunks.append(result)

        # Склеиваем
        formatted_text = "\n\n".join(formatted_chunks)
        final_length = len(formatted_text)

        logger.info(
            f"[TelegraphFormatter] Склеено: {final_length} chars "
            f"({final_length/original_length:.0%} от оригинала)"
        )

        # Верификация
        if final_length > MAX_TELEGRAPH_CHARS:
            logger.warning(
                f"[TelegraphFormatter] Превышение: {final_length} > {MAX_TELEGRAPH_CHARS}, обрезка"
            )
            formatted_text = self._truncate_smart(formatted_text, MAX_TELEGRAPH_CHARS)
            final_length = len(formatted_text)
            logger.info(f"[TelegraphFormatter] После обрезки: {final_length} chars")

        # Верификация: незакрытые code blocks
        formatted_text = self._fix_unclosed_code_blocks(formatted_text)

        compression = final_length / original_length if original_length > 0 else 1.0

        logger.info(
            f"[TelegraphFormatter] Готово: {original_length} -> {len(formatted_text)} chars "
            f"({compression:.0%}), {len(chunks)} чанков"
        )

        return TelegraphFormatResult(
            formatted_text=formatted_text,
            original_length=original_length,
            final_length=len(formatted_text),
            chunks_processed=len(chunks),
            compression_ratio=compression,
            was_compressed=needs_compression,
            was_formatted=True,
        )

    def _process_chunk(self, chunk: str, chunk_len: int, target_len: int,
                       needs_compression: bool) -> str:
        """Один чанк: сжатие + разметка за один вызов LLM."""

        if needs_compression:
            prompt = self.COMPRESS_AND_FORMAT_PROMPT.format(
                target_chars=target_len,
                current_chars=chunk_len,
                content=chunk,
            )
        else:
            prompt = self.FORMAT_ONLY_PROMPT.format(content=chunk)

        max_tokens = max(2048, target_len // 2 + 500)
        min_response = max(200, target_len // 4)

        try:
            result = self.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=0.15,
                min_response_length=min_response,
            )

            cleaned = self._clean_response(result)

            # Верификация чанка
            if len(cleaned) < 100:
                logger.warning(
                    f"[TelegraphFormatter] Чанк слишком короткий ({len(cleaned)}), оригинал"
                )
                return chunk

            if needs_compression and len(cleaned) > target_len * 2:
                logger.warning(
                    f"[TelegraphFormatter] Чанк не сжат ({len(cleaned)} > {target_len}*2), обрезка"
                )
                return self._truncate_smart(cleaned, int(target_len * 1.3))

            logger.info(f"[TelegraphFormatter] Чанк OK: {chunk_len} -> {len(cleaned)} chars")
            return cleaned

        except Exception as e:
            logger.warning(f"[TelegraphFormatter] Ошибка чанка: {e}, оригинал")
            return chunk

    def _split_into_chunks(self, content: str) -> List[str]:
        """Разбить по абзацам."""
        paragraphs = content.split('\n\n')
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 2 > MAX_CHUNK_SIZE:
                if current:
                    chunks.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para

        if current:
            chunks.append(current.strip())

        # Разбить слишком большие чанки по предложениям
        result = []
        for chunk in chunks:
            if len(chunk) > MAX_CHUNK_SIZE:
                result.extend(self._split_by_sentences(chunk))
            else:
                result.append(chunk)

        return result if result else [content]

    def _split_by_sentences(self, content: str) -> List[str]:
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

    def _truncate_smart(self, content: str, max_chars: int) -> str:
        """Умная обрезка: не ломать code blocks и абзацы."""
        if len(content) <= max_chars:
            return content

        truncated = content[:max_chars]

        # Не обрезать внутри code block
        if truncated.count("```") % 2 != 0:
            last_fence = truncated.rfind("```")
            if last_fence > max_chars * 0.5:
                truncated = truncated[:last_fence] + "\n```"

        # По абзацу
        last_para = truncated.rfind("\n\n")
        if last_para > max_chars * 0.7:
            return truncated[:last_para]

        # По предложению
        last_dot = truncated.rfind(".")
        if last_dot > max_chars * 0.7:
            return truncated[:last_dot + 1]

        return truncated

    @staticmethod
    def _fix_unclosed_code_blocks(text: str) -> str:
        """Закрыть незакрытые code blocks."""
        count = text.count("```")
        if count % 2 != 0:
            text += "\n```"
        return text

    def _clean_response(self, text: str) -> str:
        if not text:
            return ""
        result = text.strip()
        prefixes = [
            r'^Вот\s+(сокращённый|размеченный|отформатированный)\s+текст:?\s*\n*',
            r'^Результат:?\s*\n*',
            r'^РЕЗУЛЬТАТ:?\s*\n*',
            r'^РАЗМЕЧЕННЫЙ ТЕКСТ:?\s*\n*',
        ]
        for p in prefixes:
            result = re.sub(p, '', result, flags=re.IGNORECASE)
        return result.strip()

    def process(self, content: str) -> TelegraphFormatResult:
        return self.format_for_telegraph(content)
