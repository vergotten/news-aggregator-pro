# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/telegram_formatter_agent.py
# =============================================================================
"""
Агент форматирования для Telegram v10.1

Формирует ПОЛНОЦЕННЫЙ пост для Telegram-канала (до 4096 символов):
- Заголовок (bold)
- Основной текст (сжатый, но содержательный)
- Ссылка "Читать полностью →" на Telegraph
- Хештеги

Изменения v10.1:
- Улучшенное HTML-форматирование (bold ключевые моменты)
- LLM возвращает текст с **bold** маркерами → конвертируем в <b>
- Ссылка на Telegraph всегда внизу
- Чистый escape без потери форматирования

Лимиты Telegram:
- Текстовый пост: до 4096 символов
- Пост с картинкой (caption): до 1024 — НЕ используем caption
"""

import logging
import re
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

from src.application.ai_services.agents.base_agent import BaseAgent, TaskType
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)

# Лимиты Telegram
TELEGRAM_MAX_LENGTH = 4096
# Резерв на заголовок (~100), ссылки (~200), хештеги (~100), отступы (~100)
OVERHEAD_CHARS = 500
MAX_CONTENT_CHARS = TELEGRAM_MAX_LENGTH - OVERHEAD_CHARS


class TelegramPost(BaseModel):
    """Пост для Telegram-канала."""
    text: str = Field(description="Текст поста (HTML)")
    format_type: Literal["html"] = Field(default="html")
    preview_mode: bool = Field(default=False)
    telegraph_needed: bool = Field(default=True)
    telegraph_content: Optional[str] = Field(default=None)
    telegraph_url: Optional[str] = Field(default=None)
    hashtags: list[str] = Field(default_factory=list)
    cover_image: Optional[str] = Field(default=None)
    all_images: list[str] = Field(default_factory=list)
    send_image_separately: bool = Field(default=False)

    @field_validator('text')
    @classmethod
    def validate_length(cls, v: str) -> str:
        if len(v) > TELEGRAM_MAX_LENGTH:
            raise ValueError(f"Пост слишком длинный: {len(v)}")
        return v


class TelegramFormatterAgent(BaseAgent):
    """
    Агент форматирования для Telegram v10.1

    Создаёт полноценный пост (до 4096 символов).
    В конце — ссылка на Telegraph с полной версией + картинками.
    """

    agent_name = "telegram_formatter"
    task_type = TaskType.MEDIUM
    MIN_RESPONSE_LENGTH = 200

    SYSTEM_PROMPT = """Ты — редактор технического канала в Telegram.
Сожми статью в пост до {max_chars} символов.

ПРАВИЛА:
- Передай основную мысль и ключевые выводы
- Пиши как самостоятельный пост, НЕ как пересказ
- БЕЗ "В статье рассказывается...", "Автор описывает..."
- БЕЗ "я", "мы", "мой" — безличные конструкции
- Сохраняй технические термины и названия
- Выдели **ключевые моменты** жирным (через двойные звёздочки)
- Абзацы через пустую строку
- Язык: тот же что у статьи
- НЕ добавляй заголовок, ссылки, хештеги — только текст поста"""

    COMPRESS_PROMPT = """Сожми текст в пост для Telegram ({max_chars} символов максимум).

Текст должен быть самостоятельным — читатель получит основную мысль прямо в Telegram.
Выдели **ключевые моменты** жирным.

ТЕКСТ:
{content}

Напиши ТОЛЬКО текст поста (без заголовка, без ссылок):"""

    def __init__(
            self,
            llm_provider: Optional[LLMProvider] = None,
            config: Optional[ModelsConfig] = None,
            default_author: str = "TechNews",
            add_source_link: bool = True,
            **kwargs,
    ):
        super().__init__(
            llm_provider=llm_provider,
            config=config,
            max_retries=2,
            retry_delay=2.0,
        )
        self.default_author = default_author
        self.add_source_link = add_source_link
        logger.info("[INIT] TelegramFormatterAgent v10.1")

    # =================================================================
    # Публичный API
    # =================================================================

    def format_for_telegram(
            self,
            title: str,
            content: str,
            source_url: Optional[str] = None,
            tags: Optional[list[str]] = None,
            author: Optional[str] = None,
            images: Optional[list[str]] = None,
            teaser: Optional[str] = None,
            telegraph_url: Optional[str] = None,
            source_name: Optional[str] = None,
    ) -> TelegramPost:
        """
        Сформировать полноценный пост для Telegram.

        Пост = содержательный текст + ссылка на Telegraph.
        Полная версия → Telegraph (с картинками и форматированием).
        """
        hashtags = self._make_hashtags(tags or [])
        cover_image = images[0] if images else None
        has_image = cover_image is not None

        available = self._calculate_available_chars(
            title=title,
            telegraph_url=telegraph_url,
            source_url=source_url,
            source_name=source_name,
            hashtags=hashtags,
        )

        # Получаем текст поста
        post_body = self._prepare_post_body(content, teaser, available)

        # Собираем HTML пост
        post_text = self._build_telegram_post(
            title=title,
            body=post_body,
            telegraph_url=telegraph_url,
            source_url=source_url,
            source_name=source_name,
            hashtags=hashtags,
        )

        # Обрезка если > 4096
        if len(post_text) > TELEGRAM_MAX_LENGTH:
            post_text = self._safe_truncate(post_text)

        # Полный контент для Telegraph (plain text)
        telegraph_content = self._make_telegraph_text(content)

        logger.info(
            f"[Formatter] Telegram: {len(post_text)} chars "
            f"(body: {len(post_body)}), "
            f"Telegraph: {len(telegraph_content)} chars, "
            f"image: {'yes' if has_image else 'no'}"
        )

        return TelegramPost(
            text=post_text,
            format_type="html",
            preview_mode=False,
            telegraph_needed=True,
            telegraph_content=telegraph_content,
            telegraph_url=telegraph_url,
            hashtags=hashtags,
            cover_image=cover_image,
            all_images=images or [],
            send_image_separately=has_image,
        )

    def process(self, title: str, content: str, **kwargs) -> TelegramPost:
        """Основной метод для orchestrator."""
        return self.format_for_telegram(title, content, **kwargs)

    # =================================================================
    # Подготовка тела поста
    # =================================================================

    def _calculate_available_chars(
            self,
            title: str,
            telegraph_url: Optional[str],
            source_url: Optional[str],
            source_name: Optional[str],
            hashtags: list[str],
    ) -> int:
        """Рассчитать сколько символов доступно для контента."""
        used = 0
        used += len(title) + 15  # "📰 <b>title</b>\n\n"
        used += 60               # "📖 Читать полностью →"
        if source_url:
            used += len(source_name or "Источник") + 40
        if hashtags:
            used += sum(len(h) for h in hashtags[:5]) + len(hashtags[:5]) + 2

        available = TELEGRAM_MAX_LENGTH - used - 50
        return max(500, min(available, MAX_CONTENT_CHARS))

    def _prepare_post_body(
            self,
            content: str,
            teaser: Optional[str],
            max_chars: int,
    ) -> str:
        """
        Подготовить текст поста.

        Короткий контент → целиком.
        Длинный → сжатие через LLM.
        Fallback → первые абзацы.
        """
        if not content:
            return teaser or ""

        clean = self._strip_formatting(content)

        if len(clean) <= max_chars:
            logger.info(f"[Formatter] Content fits: {len(clean)} <= {max_chars}")
            return clean

        logger.info(
            f"[Formatter] Content too long ({len(clean)} chars), "
            f"compressing to {max_chars} via LLM..."
        )

        try:
            compressed = self._compress_with_llm(clean, max_chars)
            if compressed and len(compressed) >= 200:
                if len(compressed) > max_chars:
                    compressed = self._truncate_to_sentence(compressed, max_chars)
                logger.info(
                    f"[Formatter] LLM compressed: {len(clean)} → {len(compressed)} chars"
                )
                return compressed
        except Exception as e:
            logger.warning(f"[Formatter] LLM compression failed: {e}")

        logger.info("[Formatter] Using paragraph extraction fallback")
        return self._extract_paragraphs(clean, max_chars)

    def _compress_with_llm(self, content: str, max_chars: int) -> str:
        """Сжать контент через LLM."""
        input_content = content[:12000]

        system = self.SYSTEM_PROMPT.format(max_chars=max_chars)
        prompt = self.COMPRESS_PROMPT.format(
            max_chars=max_chars,
            content=input_content,
        )

        response = self.generate(
            prompt=prompt,
            system_prompt=system,
            max_tokens=max_chars // 3,
            min_response_length=200,
        )

        return self._clean_llm_response(response)

    def _extract_paragraphs(self, content: str, max_chars: int) -> str:
        """Fallback: взять первые абзацы до лимита."""
        paragraphs = content.split('\n\n')
        result_parts = []
        total = 0

        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 20:
                continue
            if para.startswith('#'):
                para = re.sub(r'^#+\s*', '', para)
            if total + len(para) + 2 > max_chars:
                break
            result_parts.append(para)
            total += len(para) + 2

        return '\n\n'.join(result_parts)

    # =================================================================
    # Сборка поста
    # =================================================================

    def _build_telegram_post(
            self,
            title: str,
            body: str,
            telegraph_url: Optional[str],
            source_url: Optional[str],
            source_name: Optional[str],
            hashtags: list[str],
    ) -> str:
        """
        Собрать пост для Telegram.

        Формат:
            📰 <b>Заголовок</b>

            Полноценный текст поста с <b>ключевыми</b> моментами...

            📖 <a href="...">Читать полностью →</a>

            #теги
        """
        parts = []

        # Заголовок
        parts.append(f"📰 <b>{_escape_html(title)}</b>")

        # Основной текст с форматированием
        if body:
            formatted_body = _markdown_to_telegram_html(body)
            parts.append(f"\n{formatted_body}")

        # Ссылка на Telegraph
        if telegraph_url:
            parts.append(
                f'\n📖 <a href="{telegraph_url}">Читать полностью →</a>'
            )
        else:
            # Плейсхолдер — заменится через inject_telegraph_url()
            parts.append("\n📖 Читать полностью → {TELEGRAPH_URL}")

        # Хештеги
        if hashtags:
            parts.append('\n' + ' '.join(hashtags[:5]))

        return '\n'.join(parts)

    # =================================================================
    # Telegraph
    # =================================================================

    def _make_telegraph_text(self, content: str) -> str:
        """Plain text для Telegraph — убираем HTML/Markdown."""
        if not content:
            return ""

        text = content
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'```[\s\S]*?```', '', text)

        paragraphs = text.split('\n\n')
        clean = []
        for para in paragraphs:
            para = para.strip()
            if para:
                para = re.sub(r'\s+', ' ', para)
                clean.append(para)

        return '\n\n'.join(clean)

    # =================================================================
    # Вспомогательные
    # =================================================================

    def inject_telegraph_url(self, post: TelegramPost, telegraph_url: str) -> TelegramPost:
        """Подставить Telegraph URL в готовый пост."""
        updated_text = post.text.replace(
            "📖 Читать полностью → {TELEGRAPH_URL}",
            f'📖 <a href="{telegraph_url}">Читать полностью →</a>'
        )

        return TelegramPost(
            text=updated_text,
            format_type=post.format_type,
            preview_mode=False,
            telegraph_needed=False,
            telegraph_content=post.telegraph_content,
            telegraph_url=telegraph_url,
            hashtags=post.hashtags,
            cover_image=post.cover_image,
            all_images=post.all_images,
            send_image_separately=post.send_image_separately,
        )

    def _safe_truncate(self, text: str) -> str:
        """Безопасная обрезка до 4096 символов."""
        if len(text) <= TELEGRAM_MAX_LENGTH:
            return text

        truncated = text[:TELEGRAM_MAX_LENGTH - 50]
        last_para = truncated.rfind('\n\n')
        if last_para > TELEGRAM_MAX_LENGTH * 0.5:
            truncated = truncated[:last_para]
        else:
            last_sentence = truncated.rfind('.')
            if last_sentence > TELEGRAM_MAX_LENGTH * 0.5:
                truncated = truncated[:last_sentence + 1]

        return truncated

    def _truncate_to_sentence(self, text: str, max_chars: int) -> str:
        """Обрезать по последнему предложению."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.6:
            return truncated[:last_period + 1]
        return truncated.rstrip()

    def _strip_formatting(self, text: str) -> str:
        """Убрать markdown/html форматирование."""
        t = re.sub(r'<[^>]+>', '', text)
        t = re.sub(r'^#{1,6}\s+', '', t, flags=re.MULTILINE)
        t = re.sub(r'\*\*([^*]+)\*\*', r'\1', t)
        t = re.sub(r'\*([^*]+)\*', r'\1', t)
        t = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', t)
        t = re.sub(r'`([^`]+)`', r'\1', t)
        t = re.sub(r'```[\s\S]*?```', '', t)
        return t.strip()

    def _clean_llm_response(self, response: str) -> str:
        """Убрать артефакты LLM."""
        text = response.strip()

        prefixes = [
            r'^Вот\s+(сжатый\s+)?текст\s*(поста)?:?\s*',
            r'^Текст\s+поста:?\s*',
            r'^Пост:?\s*',
            r'^Результат:?\s*',
            r'^Here\s+is\s+',
        ]
        for pattern in prefixes:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        text = text.strip('"').strip("'").strip('`')
        text = re.sub(r'^```.*?\n', '', text)
        text = re.sub(r'\n```$', '', text)

        # НЕ убираем ** — они будут конвертированы в <b>
        return text.strip()

    def _escape(self, text: str) -> str:
        """Экранирование HTML для Telegram."""
        return _escape_html(text)

    def _make_hashtags(self, tags: list[str], max_count: int = 5) -> list[str]:
        """Создать хештеги из тегов."""
        hashtags = []
        for tag in tags[:max_count]:
            clean = re.sub(r'[^\w\s-]', '', tag)
            clean = clean.replace(' ', '_').replace('-', '_')
            if clean and len(clean) > 1:
                hashtags.append(f"#{clean}")
        return hashtags


# =============================================================================
# Утилиты форматирования
# =============================================================================

def _escape_html(text: str) -> str:
    """Экранирование для Telegram HTML."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _markdown_to_telegram_html(text: str) -> str:
    """
    Конвертация текста с markdown-маркерами в Telegram HTML.

    Поддерживает:
    - **bold** → <b>bold</b>
    - *italic* → <i>italic</i>
    - `code` → <code>code</code>
    - Обычный текст → escape HTML

    Telegram поддерживает: <b>, <i>, <u>, <s>, <code>, <pre>, <a>.
    """
    if not text:
        return ""

    # Разбиваем по абзацам и обрабатываем каждый
    paragraphs = text.split('\n\n')
    result_parts = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        result_parts.append(_format_paragraph(para))

    return '\n\n'.join(result_parts)


def _format_paragraph(text: str) -> str:
    """
    Форматировать один абзац: escape HTML, но сохранить **bold**, *italic*, `code`.

    Логика: находим все markdown-маркеры, escape всё остальное,
    заменяем маркеры на HTML теги.
    """
    # Шаг 1: Собираем все inline-маркеры с позициями
    # Порядок важен: ** перед *, чтобы ** не перехватывался как *
    markers = []

    # **bold**
    for m in re.finditer(r'\*\*(.+?)\*\*', text):
        markers.append((m.start(), m.end(), 'b', m.group(1)))

    # Собираем занятые диапазоны
    used_ranges = [(s, e) for s, e, _, _ in markers]

    # *italic* — но не внутри **
    for m in re.finditer(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', text):
        if not any(s <= m.start() < e for s, e in used_ranges):
            markers.append((m.start(), m.end(), 'i', m.group(1)))

    # `code`
    for m in re.finditer(r'`([^`]+)`', text):
        if not any(s <= m.start() < e for s, e in used_ranges):
            markers.append((m.start(), m.end(), 'code', m.group(1)))

    # Если нет маркеров — просто escape
    if not markers:
        return _escape_html(text)

    # Шаг 2: Сортируем по позиции
    markers.sort(key=lambda x: x[0])

    # Шаг 3: Собираем результат
    result = []
    pos = 0

    for start, end, tag, content in markers:
        # Текст до маркера — escape
        if pos < start:
            result.append(_escape_html(text[pos:start]))

        # Маркер → HTML тег (контент внутри тоже escape)
        result.append(f"<{tag}>{_escape_html(content)}</{tag}>")
        pos = end

    # Хвост
    if pos < len(text):
        result.append(_escape_html(text[pos:]))

    return ''.join(result)