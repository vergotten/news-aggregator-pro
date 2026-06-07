# -*- coding: utf-8 -*-
"""
Telegraph Integration Example
==============================

Пример интеграции TelegramFormatterAgent с Telegraph API
для публикации длинных статей.

Требования:
-----------
pip install telegraph

Использование:
-------------
from telegraph_integration import TelegraphPublisher

publisher = TelegraphPublisher(
    author_name="TechNews",
    author_url="https://t.me/technews"
)

# Публикация
telegraph_url = publisher.publish(
    title="Article Title",
    content_html="<p>HTML content...</p>",
    author="Author Name"
)
"""

import logging
from typing import Optional
from telegraph import Telegraph
from telegraph.exceptions import TelegraphException

logger = logging.getLogger(__name__)


class TelegraphPublisher:
    """
    Публикация контента в Telegraph.
    
    Telegraph - платформа для длинных текстов от Telegram.
    Идеально подходит для публикации полных версий статей.
    
    Особенности:
    - Не требует аккаунта (создаётся автоматически)
    - Поддерживает HTML разметку
    - Быстрый и надёжный
    - Бесплатный
    
    Пример:
        >>> publisher = TelegraphPublisher(author_name="TechBlog")
        >>> url = publisher.publish(
        ...     title="My Article",
        ...     content_html="<p>Content here</p>"
        ... )
        >>> print(url)
        https://telegra.ph/My-Article-01-27
    """
    
    def __init__(
        self,
        author_name: str = "Anonymous",
        author_url: Optional[str] = None,
        short_name: Optional[str] = None
    ):
        """
        Инициализация Telegraph publisher.
        
        Аргументы:
            author_name: Имя автора (отображается в статье)
            author_url: URL автора (обычно ссылка на Telegram канал)
            short_name: Короткое имя (для создания аккаунта)
        """
        self.author_name = author_name
        self.author_url = author_url
        self.short_name = short_name or author_name[:32]
        
        self._telegraph = None
        self._initialize_account()
    
    def _initialize_account(self):
        """Инициализация Telegraph аккаунта."""
        try:
            self._telegraph = Telegraph()
            
            # Создаём аккаунт (или получаем существующий)
            response = self._telegraph.create_account(
                short_name=self.short_name,
                author_name=self.author_name,
                author_url=self.author_url
            )
            
            logger.info(
                f"Telegraph account initialized: {response.get('short_name')}"
            )
            
        except TelegraphException as e:
            logger.error(f"Failed to initialize Telegraph account: {e}")
            raise
    
    def publish(
        self,
        title: str,
        content_html: str,
        author: Optional[str] = None
    ) -> str:
        """
        Публикация статьи в Telegraph.
        
        Аргументы:
            title: Заголовок статьи
            content_html: HTML контент
            author: Автор (переопределяет default)
            
        Возвращает:
            URL опубликованной статьи
        """
        try:
            # Публикация страницы
            response = self._telegraph.create_page(
                title=title,
                html_content=content_html,
                author_name=author or self.author_name,
                author_url=self.author_url
            )
            
            url = f"https://telegra.ph/{response['path']}"
            logger.info(f"Published to Telegraph: {url}")
            
            return url
            
        except TelegraphException as e:
            logger.error(f"Failed to publish to Telegraph: {e}")
            raise
    
    def edit(
        self,
        path: str,
        title: str,
        content_html: str
    ) -> str:
        """
        Редактирование существующей статьи.
        
        Аргументы:
            path: Path статьи (из URL)
            title: Новый заголовок
            content_html: Новый контент
            
        Возвращает:
            URL обновлённой статьи
        """
        try:
            response = self._telegraph.edit_page(
                path=path,
                title=title,
                html_content=content_html,
                author_name=self.author_name,
                author_url=self.author_url
            )
            
            url = f"https://telegra.ph/{response['path']}"
            logger.info(f"Updated Telegraph page: {url}")
            
            return url
            
        except TelegraphException as e:
            logger.error(f"Failed to edit Telegraph page: {e}")
            raise


# =============================================================================
# Интеграция с TelegramFormatterAgent
# =============================================================================

def publish_article_to_telegram_and_telegraph(
    title: str,
    content: str,
    source_url: Optional[str] = None,
    tags: Optional[list[str]] = None,
    telegram_bot = None,
    channel_id: str = None
):
    """
    Полный пайплайн: форматирование + публикация в Telegram/Telegraph.
    
    Аргументы:
        title: Заголовок статьи
        content: Полный текст статьи
        source_url: URL источника
        tags: Теги статьи
        telegram_bot: Telegram Bot instance
        channel_id: ID канала для публикации
        
    Возвращает:
        dict с результатами публикации
    """
    from src.application.ai_services.agents import TelegramFormatterAgent
    
    # 1. Форматирование для Telegram
    telegram_agent = TelegramFormatterAgent(
        default_author="TechNews",
        add_source_link=True
    )
    
    post = telegram_agent.format_for_telegram(
        title=title,
        content=content,
        source_url=source_url,
        tags=tags
    )
    
    result = {
        'telegram_posted': False,
        'telegraph_url': None,
        'error': None
    }
    
    try:
        # 2. Если нужен Telegraph - публикуем
        if post.telegraph_needed:
            logger.info("Publishing full version to Telegraph")
            
            publisher = TelegraphPublisher(
                author_name="TechNews",
                author_url="https://t.me/technews"
            )
            
            telegraph_url = publisher.publish(
                title=title,
                content_html=post.telegraph_content
            )
            
            result['telegraph_url'] = telegraph_url
            
            # Заменяем placeholder в Telegram посте
            final_telegram_text = post.text.replace(
                "{TELEGRAPH_URL}",
                telegraph_url
            )
        else:
            final_telegram_text = post.text
        
        # 3. Публикация в Telegram
        if telegram_bot and channel_id:
            logger.info("Publishing to Telegram channel")
            
            telegram_bot.send_message(
                chat_id=channel_id,
                text=final_telegram_text,
                parse_mode="HTML",
                disable_web_page_preview=False
            )
            
            result['telegram_posted'] = True
        
        logger.info("Article published successfully")
        return result
        
    except Exception as e:
        logger.error(f"Publishing failed: {e}", exc_info=True)
        result['error'] = str(e)
        return result


# =============================================================================
# Пример использования
# =============================================================================

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    # Пример 1: Простая публикация в Telegraph
    print("\n=== Пример 1: Публикация в Telegraph ===")
    
    publisher = TelegraphPublisher(
        author_name="TechNews",
        author_url="https://t.me/technews"
    )
    
    telegraph_url = publisher.publish(
        title="Python 3.13 Released: What's New",
        content_html="""
            <h3>Introduction</h3>
            <p>Python 3.13 brings significant performance improvements...</p>
            
            <h3>New Features</h3>
            <ul>
                <li>JIT Compilation - 40% faster execution</li>
                <li>Improved error messages</li>
                <li>Better type hints support</li>
            </ul>
            
            <h3>Conclusion</h3>
            <p>This release marks a major milestone...</p>
        """
    )
    
    print(f"✅ Published: {telegraph_url}")
    
    # Пример 2: Полный пайплайн (требует Telegram bot)
    print("\n=== Пример 2: Полный пайплайн ===")
    print("(Для запуска нужен настроенный Telegram bot)")
    
    """
    from telegram import Bot
    
    bot = Bot(token="YOUR_BOT_TOKEN")
    
    result = publish_article_to_telegram_and_telegraph(
        title="Python 3.13 Released",
        content="Very long article content..." * 100,
        source_url="https://python.org/release",
        tags=["python", "release", "performance"],
        telegram_bot=bot,
        channel_id="@your_channel"
    )
    
    if result['telegram_posted']:
        print(f"✅ Posted to Telegram")
        if result['telegraph_url']:
            print(f"✅ Full version: {result['telegraph_url']}")
    """
    
    # Пример 3: Использование с TelegramFormatterAgent
    print("\n=== Пример 3: С TelegramFormatterAgent ===")
    
    from src.application.ai_services.agents import TelegramFormatterAgent
    
    agent = TelegramFormatterAgent()
    
    # Длинная статья
    long_content = """
    Это очень длинная статья, которая не поместится в Telegram.
    """ * 500  # ~10k символов
    
    post = agent.format_for_telegram(
        title="Long Article Title",
        content=long_content,
        tags=["tech", "news"]
    )
    
    if post.telegraph_needed:
        print("📝 Статья длинная, нужен Telegraph")
        
        # Публикуем в Telegraph
        telegraph_url = publisher.publish(
            title="Long Article Title",
            content_html=post.telegraph_content
        )
        
        # Обновляем Telegram пост
        final_text = post.text.replace("{TELEGRAPH_URL}", telegraph_url)
        
        print(f"✅ Telegram preview готов ({len(final_text)} chars)")
        print(f"✅ Telegraph URL: {telegraph_url}")
    else:
        print("✅ Статья короткая, помещается в Telegram")
        print(f"   Длина: {len(post.text)} chars")
