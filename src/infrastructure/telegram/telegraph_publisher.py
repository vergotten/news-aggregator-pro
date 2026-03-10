# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/infrastructure/telegram/telegraph_publisher.py
# =============================================================================
"""
Telegraph Publisher Service v3.0

Создание страниц на Telegraph через JSON API (Node format).

Ключевые изменения v3.0:
- Используется JSON node format (как в Telegraph API напрямую)
  вместо html_content через python-telegraph библиотеку
- Умное распределение изображений между абзацами
- Поддержка: заголовки, списки, код, цитаты, ссылки
- Работает с plain text (выход StyleNormalizerAgent)

Зависимости:
    pip install requests
"""

import os
import re
import json
import logging
from typing import Optional, List, Dict, Any

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# Конфигурация
# =============================================================================

class TelegraphConfig:
    """Конфигурация Telegraph."""

    def __init__(
        self,
        short_name: str = "NewsAggregator",
        author_name: str = "News Aggregator Bot",
        author_url: str = "",
        max_title_length: int = 256,
        max_content_length: int = 64000,
        paragraphs_per_image: int = 3,
    ):
        self.short_name = short_name
        self.author_name = author_name
        self.author_url = author_url or os.getenv("TELEGRAM_CHANNEL_URL", "")
        self.max_title_length = max_title_length
        self.max_content_length = max_content_length
        self.paragraphs_per_image = paragraphs_per_image


class TelegraphResult:
    """Результат создания страницы."""

    def __init__(
        self,
        success: bool,
        url: Optional[str] = None,
        error: Optional[str] = None,
        title: Optional[str] = None,
    ):
        self.success = success
        self.url = url
        self.error = error
        self.title = title


# =============================================================================
# Telegraph Publisher v3.0
# =============================================================================

TELEGRAPH_API = "https://api.telegra.ph"


class TelegraphPublisher:
    """
    Сервис создания страниц на Telegraph.

    Использует JSON Node format (напрямую через API),
    что даёт более надёжное форматирование чем html_content.

    Поддерживает:
    - Обложка (первое изображение перед текстом)
    - Изображения между абзацами (через каждые N абзацев)
    - Оставшиеся изображения в конце
    - Заголовки (## markdown и короткие строки)
    - Блоки кода (``` ... ```)
    - Списки (- item, * item, 1. item)
    - Цитаты (> text)
    - Inline форматирование (**bold**, *italic*, `code`)
    - Ссылки [text](url)
    """

    def __init__(self, config: Optional[TelegraphConfig] = None):
        self.config = config or TelegraphConfig()
        self._access_token: Optional[str] = None
        logger.info("[Telegraph] TelegraphPublisher v3.0 initialized")

    # -----------------------------------------------------------------
    # Аккаунт
    # -----------------------------------------------------------------

    def _ensure_account(self) -> str:
        """Создать аккаунт Telegraph и вернуть access_token."""
        if self._access_token:
            return self._access_token

        resp = requests.post(f"{TELEGRAPH_API}/createAccount", data={
            "short_name": self.config.short_name,
            "author_name": self.config.author_name,
        })
        resp.raise_for_status()
        result = resp.json()

        if not result.get("ok"):
            raise RuntimeError(f"Telegraph createAccount failed: {result}")

        self._access_token = result["result"]["access_token"]
        logger.info("[Telegraph] Аккаунт создан: %s", self.config.short_name)
        return self._access_token

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
        """
        Создать страницу на Telegraph.

        Args:
            title: Заголовок страницы
            content: Текст статьи (plain text или markdown)
            images: URL изображений для вставки
            author_name: Имя автора
            source_url: Ссылка на оригинал

        Returns:
            TelegraphResult
        """
        try:
            token = self._ensure_account()

            clean_title = title[:self.config.max_title_length].strip()

            # Конвертируем контент в Telegraph JSON nodes
            nodes = self._content_to_nodes(content, images, source_url)

            # Публикуем через API
            resp = requests.post(f"{TELEGRAPH_API}/createPage", data={
                "access_token": token,
                "title": clean_title,
                "author_name": author_name or self.config.author_name,
                "content": json.dumps(nodes, ensure_ascii=False),
                "return_content": "false",
            })
            resp.raise_for_status()
            result = resp.json()

            if not result.get("ok"):
                error = result.get("error", "Unknown Telegraph error")
                logger.error("[Telegraph] API ошибка: %s", error)
                return TelegraphResult(success=False, error=error, title=clean_title)

            url = result["result"].get("url")
            logger.info("[Telegraph] Страница создана: %s", url)
            return TelegraphResult(success=True, url=url, title=clean_title)

        except Exception as e:
            logger.error("[Telegraph] Ошибка: %s", e)
            return TelegraphResult(success=False, error=str(e), title=title[:60])

    # -----------------------------------------------------------------
    # Конвертация контента → Telegraph JSON nodes
    # -----------------------------------------------------------------

    def _content_to_nodes(
        self,
        content: str,
        images: Optional[List[str]] = None,
        source_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Конвертировать текст в Telegraph JSON nodes.

        Telegraph Node format:
            {"tag": "p", "children": ["text"]}
            {"tag": "h4", "children": ["heading"]}
            {"tag": "figure", "children": [{"tag": "img", "attrs": {"src": "..."}}]}
        """
        if not content:
            return [{"tag": "p", "children": ["Контент отсутствует"]}]

        images = images or []
        remaining_images = list(images)
        nodes = []

        # Обложка — первое изображение перед текстом
        if remaining_images:
            nodes.append(_make_image_node(remaining_images.pop(0)))

        # Разбиваем контент на блоки
        blocks = self._split_into_blocks(content)

        paragraph_count = 0

        for block in blocks:
            block_type, block_content = block

            if block_type == "code":
                lang, code = block_content
                nodes.append(_make_code_node(code))

            elif block_type == "heading":
                level, text = block_content
                tag = "h3" if level <= 2 else "h4"
                nodes.append({"tag": tag, "children": [text]})

            elif block_type == "list":
                nodes.append(_make_list_node(block_content))

            elif block_type == "quote":
                nodes.append({"tag": "blockquote", "children": [block_content]})

            elif block_type == "paragraph":
                # Парсим inline форматирование
                children = _parse_inline(block_content)
                nodes.append({"tag": "p", "children": children})
                paragraph_count += 1

                # Вставляем изображение каждые N абзацев
                if (remaining_images
                        and paragraph_count % self.config.paragraphs_per_image == 0):
                    nodes.append(_make_image_node(remaining_images.pop(0)))

        # Оставшиеся изображения
        if remaining_images:
            nodes.append({"tag": "hr"})
            for img_url in remaining_images[:5]:
                nodes.append(_make_image_node(img_url))

        # Ссылка на оригинал — убрана, трафик идёт в Telegram канал

        return nodes

    # -----------------------------------------------------------------
    # Парсинг блоков
    # -----------------------------------------------------------------

    def _split_into_blocks(self, content: str) -> List:
        """
        Разбить контент на типизированные блоки.

        Returns:
            Список кортежей: (тип, контент)
            Типы: "code", "heading", "list", "quote", "paragraph"
        """
        blocks = []

        # Сначала извлекаем code blocks (``` ... ```)
        code_pattern = re.compile(r'```(\w*)\n?(.*?)```', re.DOTALL)
        last_end = 0

        for match in code_pattern.finditer(content):
            before = content[last_end:match.start()].strip()
            if before:
                blocks.extend(self._parse_text_blocks(before))

            lang = match.group(1) or ""
            code = match.group(2).strip()
            blocks.append(("code", (lang, code)))

            last_end = match.end()

        remaining = content[last_end:].strip()
        if remaining:
            blocks.extend(self._parse_text_blocks(remaining))

        return blocks

    def _parse_text_blocks(self, text: str) -> List:
        """
        Парсит текст (без code blocks) на блоки.

        Распознаёт: заголовки, списки, цитаты, параграфы.
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

            # Маркированный список
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

            # Нумерованный список
            num_pattern = re.compile(r'^\s*\d+[.)]\s+(.+)$')
            if all(num_pattern.match(line) for line in lines if line.strip()):
                items = []
                for line in lines:
                    m = num_pattern.match(line)
                    if m:
                        items.append(m.group(1).strip())
                if items:
                    blocks.append(("list", items))
                    continue

            # Цитата
            if para.startswith(">"):
                quote_text = re.sub(r'^>\s*', '', para, flags=re.MULTILINE)
                blocks.append(("quote", quote_text.strip()))
                continue

            # Inline code block (отступ 4+ пробела)
            if all(line.startswith("    ") or not line.strip() for line in lines):
                code = "\n".join(
                    line[4:] if line.startswith("    ") else line
                    for line in lines
                )
                blocks.append(("code", ("", code.strip())))
                continue

            # Короткая строка без финальной пунктуации → подзаголовок
            if (len(para) < 80
                    and "\n" not in para
                    and not para.endswith((".", ":", "!", "?", ","))):
                clean = re.sub(r'\*\*(.+?)\*\*', r'\1', para)
                blocks.append(("heading", (3, clean)))
                continue

            # Обычный параграф
            blocks.append(("paragraph", para))

        return blocks


# =============================================================================
# Создание Telegraph JSON nodes
# =============================================================================

def _make_image_node(url: str, caption: Optional[str] = None) -> Dict:
    """Создать node для изображения с <figure>."""
    children = [{"tag": "img", "attrs": {"src": url}}]
    if caption:
        children.append({"tag": "figcaption", "children": [caption]})
    return {"tag": "figure", "children": children}


def _make_code_node(code: str) -> Dict:
    """Создать node для блока кода <pre><code>."""
    return {
        "tag": "pre",
        "children": [{"tag": "code", "children": [code]}]
    }


def _make_list_node(items: List[str]) -> Dict:
    """Создать node для списка <ul><li>."""
    return {
        "tag": "ul",
        "children": [
            {"tag": "li", "children": _parse_inline(item)}
            for item in items
        ]
    }


def _parse_inline(text: str) -> List:
    """
    Парсить inline форматирование в тексте.

    Поддерживает:
    - **bold** → <strong>
    - *italic* → <em>
    - `code` → <code>
    - [text](url) → <a>

    Returns:
        Список children для Telegraph node (строки и теги)
    """
    if not text:
        return [""]

    # Паттерны inline-форматирования (порядок важен!)
    patterns = [
        # **bold**
        (re.compile(r'\*\*(.+?)\*\*'), "strong"),
        # *italic* (но не ** который уже обработан)
        (re.compile(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)'), "em"),
        # `inline code`
        (re.compile(r'`([^`]+)`'), "code"),
        # [text](url) → <a>
        (re.compile(r'\[([^\]]+)\]\(([^)]+)\)'), "link"),
    ]

    # Проверяем, есть ли хоть один паттерн
    has_formatting = any(p.search(text) for p, _ in patterns)
    if not has_formatting:
        return [text]

    # Собираем все match-и с позициями
    matches = []
    for pattern, tag_type in patterns:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), m, tag_type))

    if not matches:
        return [text]

    # Сортируем по позиции, убираем перекрытия
    matches.sort(key=lambda x: x[0])
    filtered = []
    last_end = 0
    for start, end, m, tag_type in matches:
        if start >= last_end:
            filtered.append((start, end, m, tag_type))
            last_end = end

    # Собираем children
    children = []
    pos = 0
    for start, end, m, tag_type in filtered:
        # Текст перед match
        if pos < start:
            children.append(text[pos:start])

        if tag_type == "link":
            link_text = m.group(1)
            link_url = m.group(2)
            children.append({
                "tag": "a",
                "attrs": {"href": link_url},
                "children": [link_text]
            })
        elif tag_type in ("strong", "em", "code"):
            children.append({
                "tag": tag_type,
                "children": [m.group(1)]
            })

        pos = end

    # Текст после последнего match
    if pos < len(text):
        children.append(text[pos:])

    return children if children else [text]