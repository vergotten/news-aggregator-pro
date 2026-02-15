# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/infrastructure/telegram/telegraph_publisher.py
# =============================================================================
"""
Telegraph Publisher Service v2.1

Создание страниц на Telegraph с правильным форматированием:
- Код оборачивается в <pre> (Telegraph рендерит как моноширинный блок)
- Изображения вставляются между абзацами (а не все в конце)
- Заголовки → <h3>/<h4>
- Списки → <ul>/<li>
- Цитаты → <blockquote>

Зависимости:
    pip install telegraph
"""

import os
import re
import logging
from typing import Optional, List
from dataclasses import dataclass

from telegraph import Telegraph

logger = logging.getLogger(__name__)


# =============================================================================
# Конфигурация
# =============================================================================

@dataclass
class TelegraphConfig:
    """Конфигурация Telegraph."""
    short_name: str = "NewsAggregator"
    author_name: str = "News Aggregator Bot"
    author_url: str = ""
    max_title_length: int = 256
    max_content_length: int = 64000

    # Сколько абзацев между вставками изображений
    paragraphs_per_image: int = 3


@dataclass
class TelegraphResult:
    """Результат создания страницы."""
    success: bool
    url: Optional[str] = None
    error: Optional[str] = None
    title: Optional[str] = None


# =============================================================================
# Telegraph Publisher
# =============================================================================

class TelegraphPublisher:
    """
    Сервис создания страниц на Telegraph.

    Умная конвертация контента:
    - Распознаёт блоки кода (```, отступы) → <pre>
    - Вставляет изображения между абзацами
    - Распознаёт заголовки (##, короткие строки) → <h3>/<h4>
    - Распознаёт списки (-, *, •) → <ul><li>
    - Распознаёт цитаты (>) → <blockquote>
    """

    def __init__(self, config: Optional[TelegraphConfig] = None):
        self.config = config or TelegraphConfig()
        self.config.author_url = os.getenv(
            "TELEGRAM_CHANNEL_URL", self.config.author_url
        )
        self._telegraph: Optional[Telegraph] = None
        self._account_created = False
        logger.info("[Telegraph] TelegraphPublisher v2.1 initialized")

    # -----------------------------------------------------------------
    # Аккаунт
    # -----------------------------------------------------------------

    def _ensure_account(self) -> Telegraph:
        if self._telegraph and self._account_created:
            return self._telegraph

        self._telegraph = Telegraph()
        self._telegraph.create_account(
            short_name=self.config.short_name,
            author_name=self.config.author_name,
            author_url=self.config.author_url or None,
        )
        self._account_created = True
        logger.info("[Telegraph] Аккаунт создан: %s", self.config.short_name)
        return self._telegraph

    # -----------------------------------------------------------------
    # Создание страницы
    # -----------------------------------------------------------------

    def create_page(
        self,
        title: str,
        content: str,
        images: Optional[List[str]] = None,
        author_name: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> TelegraphResult:
        """Создаёт страницу на Telegraph."""
        try:
            telegraph = self._ensure_account()

            clean_title = title[:self.config.max_title_length].strip()

            html_content = self._content_to_telegraph_html(
                content, images, source_url
            )

            if len(html_content) > self.config.max_content_length:
                html_content = html_content[:self.config.max_content_length]
                html_content += "<p><i>... (текст сокращён)</i></p>"

            response = telegraph.create_page(
                title=clean_title,
                html_content=html_content,
                author_name=author_name or self.config.author_name,
                author_url=self.config.author_url or None,
            )

            url = response.get("url")
            logger.info("[Telegraph] Создана страница: %s", url)

            return TelegraphResult(success=True, url=url, title=clean_title)

        except Exception as e:
            logger.error("[Telegraph] Ошибка создания страницы: %s", e)
            return TelegraphResult(success=False, error=str(e), title=title[:60])

    # -----------------------------------------------------------------
    # Конвертация контента → Telegraph HTML
    # -----------------------------------------------------------------

    def _content_to_telegraph_html(
        self,
        content: str,
        images: Optional[List[str]] = None,
        source_url: Optional[str] = None,
    ) -> str:
        """
        Умная конвертация контента в Telegraph HTML.

        Распознаёт:
        - Блоки кода (``` ... ```) → <pre>
        - Markdown заголовки (## ...) → <h3>/<h4>
        - Списки (- item, * item, • item) → <ul><li>
        - Цитаты (> text) → <blockquote>
        - Обычные параграфы → <p>

        Изображения распределяются между абзацами.
        """
        if not content:
            return "<p>Контент отсутствует</p>"

        images = images or []
        remaining_images = list(images)  # копия для вставки
        parts = []

        # Обложка — первое изображение перед текстом
        if remaining_images:
            parts.append(self._make_image(remaining_images.pop(0)))

        # Разбиваем на блоки (абзацы и code blocks)
        blocks = self._split_into_blocks(content)

        paragraph_count = 0

        for block in blocks:
            block_type, block_content = block

            if block_type == "code":
                # Блок кода → <pre>
                parts.append(self._make_code_block(block_content))

            elif block_type == "heading":
                # Заголовок → <h3> или <h4>
                level, text = block_content
                tag = "h3" if level <= 2 else "h4"
                parts.append(f"<{tag}>{_escape_html(text)}</{tag}>")

            elif block_type == "list":
                # Список → <ul><li>
                parts.append(self._make_list(block_content))

            elif block_type == "quote":
                # Цитата → <blockquote>
                parts.append(f"<blockquote>{_escape_html(block_content)}</blockquote>")

            elif block_type == "paragraph":
                # Обычный параграф → <p>
                text = _escape_html(block_content).replace("\n", "<br/>")
                parts.append(f"<p>{text}</p>")
                paragraph_count += 1

                # Вставляем изображение каждые N абзацев
                if (remaining_images
                        and paragraph_count % self.config.paragraphs_per_image == 0):
                    parts.append(self._make_image(remaining_images.pop(0)))

        # Оставшиеся изображения в конце (максимум 5)
        if remaining_images:
            parts.append("<hr/>")
            for img_url in remaining_images[:5]:
                parts.append(self._make_image(img_url))

        # Ссылка на оригинал
        if source_url:
            parts.append(
                f'<p><a href="{source_url}">📎 Читать оригинал</a></p>'
            )

        return "\n".join(parts)

    # -----------------------------------------------------------------
    # Парсинг блоков
    # -----------------------------------------------------------------

    def _split_into_blocks(self, content: str) -> list:
        """
        Разбить контент на типизированные блоки.

        Returns:
            Список кортежей: (тип, контент)
            Типы: "code", "heading", "list", "quote", "paragraph"
        """
        blocks = []

        # Сначала извлекаем code blocks (``` ... ```)
        # Разбиваем текст по code fences
        code_pattern = re.compile(r'```(\w*)\n?(.*?)```', re.DOTALL)
        last_end = 0

        for match in code_pattern.finditer(content):
            # Текст до code block
            before = content[last_end:match.start()].strip()
            if before:
                blocks.extend(self._parse_text_blocks(before))

            # Code block
            lang = match.group(1) or ""
            code = match.group(2).strip()
            blocks.append(("code", (lang, code)))

            last_end = match.end()

        # Текст после последнего code block
        remaining = content[last_end:].strip()
        if remaining:
            blocks.extend(self._parse_text_blocks(remaining))

        return blocks

    def _parse_text_blocks(self, text: str) -> list:
        """
        Парсит текст (без code blocks) на блоки.

        Распознаёт заголовки, списки, цитаты, параграфы.
        """
        blocks = []
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Markdown заголовок: ## Title
            heading_match = re.match(r'^(#{1,4})\s+(.+)$', para, re.MULTILINE)
            if heading_match and "\n" not in para:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                blocks.append(("heading", (level, heading_text)))
                continue

            # Список: строки начинающиеся с - или * или •
            lines = para.split("\n")
            list_pattern = re.compile(r'^\s*[-*•]\s+(.+)$')
            if all(list_pattern.match(line) for line in lines if line.strip()):
                items = []
                for line in lines:
                    m = list_pattern.match(line)
                    if m:
                        items.append(m.group(1).strip())
                if items:
                    blocks.append(("list", items))
                    continue

            # Нумерованный список: 1. item, 2. item
            num_list_pattern = re.compile(r'^\s*\d+[.)]\s+(.+)$')
            if all(num_list_pattern.match(line) for line in lines if line.strip()):
                items = []
                for line in lines:
                    m = num_list_pattern.match(line)
                    if m:
                        items.append(m.group(1).strip())
                if items:
                    blocks.append(("list", items))
                    continue

            # Цитата: > text
            if para.startswith(">"):
                quote_text = re.sub(r'^>\s*', '', para, flags=re.MULTILINE)
                blocks.append(("quote", quote_text.strip()))
                continue

            # Inline code block (отступ 4+ пробела на каждой строке)
            if all(line.startswith("    ") or not line.strip() for line in lines):
                code = "\n".join(line[4:] if line.startswith("    ") else line for line in lines)
                blocks.append(("code", ("", code.strip())))
                continue

            # Короткая строка без точки → заголовок (эвристика)
            if len(para) < 80 and not para.endswith((".", ":", "!", "?")):
                # Убираем markdown bold
                clean = re.sub(r'\*\*(.+?)\*\*', r'\1', para)
                blocks.append(("heading", (2, clean)))
                continue

            # Обычный параграф
            # Убираем markdown inline formatting
            clean = para
            clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)  # **bold** → bold
            clean = re.sub(r'\*(.+?)\*', r'\1', clean)        # *italic* → italic
            clean = re.sub(r'__(.+?)__', r'\1', clean)
            clean = re.sub(r'_(.+?)_', r'\1', clean)
            # Markdown ссылки [text](url) → text
            clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)
            # Inline code `code` оставляем как есть (Telegraph нет inline code)
            clean = re.sub(r'`([^`]+)`', r'\1', clean)

            blocks.append(("paragraph", clean))

        return blocks

    # -----------------------------------------------------------------
    # Рендер HTML-элементов
    # -----------------------------------------------------------------

    @staticmethod
    def _make_code_block(code_data) -> str:
        """Блок кода → <pre>."""
        if isinstance(code_data, tuple):
            lang, code = code_data
        else:
            lang, code = "", code_data

        escaped = _escape_html(code)

        # Telegraph рендерит <pre> как моноширинный блок с серым фоном
        return f"<pre>{escaped}</pre>"

    @staticmethod
    def _make_list(items: list) -> str:
        """Список → <ul><li>."""
        li_items = "\n".join(f"<li>{_escape_html(item)}</li>" for item in items)
        return f"<ul>\n{li_items}\n</ul>"

    @staticmethod
    def _make_image(url: str, caption: Optional[str] = None) -> str:
        """Изображение → <figure><img>."""
        if caption:
            return (
                f'<figure>'
                f'<img src="{url}"/>'
                f'<figcaption>{_escape_html(caption)}</figcaption>'
                f'</figure>'
            )
        return f'<figure><img src="{url}"/></figure>'


# =============================================================================
# Утилиты
# =============================================================================

def _escape_html(text: str) -> str:
    """Экранирование для Telegraph HTML."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )