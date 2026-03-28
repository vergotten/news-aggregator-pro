# -*- coding: utf-8 -*-
"""
web/app.py — FastAPI блог + RSS для Дзена

Публичный сайт:
  /                    — главная (список статей)
  /article/{slug}      — страница статьи
  /feed/dzen.xml       — RSS для Дзена
  /feed/rss.xml        — обычный RSS
  /robots.txt          — для роботов
  /api/articles        — JSON API (для админки)
  /admin               — панель администратора

Работает с любой PostgreSQL: Supabase (облако) или локальная.
Просто укажи нужный DATABASE_URL в .env.
"""

import os
import re
import html
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import psycopg2
import psycopg2.extras

# =================================================================
# Загрузка .env  (ищем сначала рядом с app.py, потом на уровень выше)
# =================================================================

for _env_path in [
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(__file__), "..", ".env"),
]:
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _, _value = _line.partition("=")
                    _key = _key.strip()
                    _value = _value.strip().strip('"').strip("'")
                    if _key and _value:
                        os.environ.setdefault(_key, _value)
        break

# =================================================================
# Конфигурация
# =================================================================

SITE_TITLE       = "НейроКотΔ"
SITE_DESCRIPTION = "Технические статьи с AI-обработкой"
SITE_URL         = os.getenv("SITE_URL", "http://localhost:8080")
SITE_LANGUAGE    = "ru"

# =================================================================
# База данных
#
# Работает с любым PostgreSQL:
#   Supabase:  DATABASE_URL=postgresql://postgres.xxx:PASS@aws-...pooler.supabase.com:6543/postgres
#   Локальная: DATABASE_URL=postgresql://user:pass@localhost:5432/news_aggregator
#   Docker:    DATABASE_URL=postgresql://newsaggregator:changeme123@postgres:5432/news_aggregator
# =================================================================

def get_db_url() -> str:
    return os.getenv("DATABASE_URL", "")


def get_conn():
    url = get_db_url()
    if not url:
        raise RuntimeError(
            "DATABASE_URL не задан. "
            "Добавь в .env строку вида:\n"
            "  DATABASE_URL=postgresql://user:pass@localhost:5432/dbname\n"
            "или для Supabase:\n"
            "  DATABASE_URL=postgresql://postgres.xxx:PASS@aws-...supabase.com:6543/postgres"
        )
    return psycopg2.connect(url)


# =================================================================
# FastAPI App
# =================================================================

app = FastAPI(title=SITE_TITLE, docs_url="/api/docs")

# Подключаем админ-панель СРАЗУ после создания app
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from admin_routes import router as admin_router  # noqa: E402
app.include_router(admin_router)

# Шаблоны и статика
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)


# =================================================================
# Утилиты
# =================================================================

def make_slug(title: str) -> str:
    """Создать ЧПУ slug из заголовка."""
    slug = title.lower().strip()
    translit = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'j', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '',
        'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    }
    result = ""
    for char in slug:
        if char in translit:
            result += translit[char]
        elif char.isalnum():
            result += char
        elif char in (' ', '-', '_'):
            result += '-'
    result = re.sub(r'-+', '-', result).strip('-')
    return result[:80]


def markdown_to_html(text: str) -> str:
    """Простая конвертация markdown → HTML."""
    if not text:
        return ""
    text = re.sub(r'```(\w*)\n(.*?)```', r'<pre><code>\2</code></pre>', text, flags=re.DOTALL)
    text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$',  r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',   r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    lines = text.split('\n')
    in_list = False
    result = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^[-*•]\s+', stripped):
            if not in_list:
                result.append('<ul>')
                in_list = True
            item = re.sub(r'^[-*•]\s+', '', stripped)
            result.append(f'<li>{item}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            result.append(line)
    if in_list:
        result.append('</ul>')
    text = '\n'.join(result)
    paragraphs = text.split('\n\n')
    processed = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith(('<h', '<pre', '<ul', '<ol', '<blockquote', '<figure')):
            processed.append(p)
        else:
            p = p.replace('\n', '<br>')
            processed.append(f'<p>{p}</p>')
    return '\n'.join(processed)


def get_articles(status: str = "published", limit: int = 20) -> List[dict]:
    """Получить статьи из БД."""
    c = get_conn()
    cur = c.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT id, title, editorial_title, editorial_teaser, editorial_rewritten,
               telegraph_content_html, telegram_cover_image, telegraph_url,
               tags, images, relevance_score, author, source, url,
               published_at, created_at, updated_at
        FROM articles
        WHERE status = %s AND relevance_score >= 5
        ORDER BY updated_at DESC
        LIMIT %s
    """, (status, limit))
    rows = cur.fetchall()
    cur.close()
    c.close()

    articles = []
    for r in rows:
        r = dict(r)
        title = r.get('editorial_title') or r.get('title') or 'Без заголовка'
        r['display_title'] = title
        r['slug'] = make_slug(title)
        r['teaser'] = r.get('editorial_teaser') or ''
        r['cover_image'] = (
            r.get('telegram_cover_image')
            or ((r.get('images') or [''])[0] if r.get('images') else '')
        )
        articles.append(r)
    return articles


def get_article_by_slug(slug: str) -> Optional[dict]:
    """Найти статью по slug."""
    articles = get_articles(limit=100)
    for a in articles:
        if a['slug'] == slug:
            return a
    return None


# =================================================================
# Хелпер: передаём в шаблоны флаг is_admin для nav-кнопки
# =================================================================

SESSION_COOKIE = "admin_session"

def _is_admin(request: Request) -> bool:
    """Проверить наличие валидной сессии (для nav-кнопки)."""
    try:
        from admin_routes import verify_session_token
        token = request.cookies.get(SESSION_COOKIE, "")
        return bool(token) and verify_session_token(token)
    except Exception:
        return False


# =================================================================
# Роуты — HTML страницы
# =================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        articles = get_articles(limit=20)
    except Exception:
        articles = []
    return templates.TemplateResponse("index.html", {
        "request":    request,
        "articles":   articles,
        "site_title": SITE_TITLE,
        "site_url":   SITE_URL,
        "is_admin":   _is_admin(request),
    })


@app.get("/article/{slug}", response_class=HTMLResponse)
async def article_page(request: Request, slug: str):
    article = get_article_by_slug(slug)
    if not article:
        raise HTTPException(status_code=404, detail="Статья не найдена")
    content_md   = article.get('telegraph_content_html') or article.get('editorial_rewritten') or ''
    content_html = markdown_to_html(content_md)
    return templates.TemplateResponse("article.html", {
        "request":      request,
        "article":      article,
        "content_html": content_html,
        "site_title":   SITE_TITLE,
        "site_url":     SITE_URL,
        "is_admin":     _is_admin(request),
    })


# =================================================================
# RSS
# =================================================================

@app.get("/feed/dzen.xml")
async def dzen_rss():
    articles = get_articles(limit=50)
    items = []
    for a in articles:
        title       = a['display_title']
        slug        = a['slug']
        article_url = f"{SITE_URL}/article/{slug}"
        teaser      = a.get('teaser') or ''
        author      = a.get('author') or 'AI Digest'
        content_md  = a.get('telegraph_content_html') or a.get('editorial_rewritten') or ''
        content_html = markdown_to_html(content_md)
        cover       = a.get('cover_image') or ''
        enclosure   = f'<enclosure url="{html.escape(cover)}" type="image/jpeg"/>' if cover else ''
        pub_date    = a.get('published_at') or a.get('updated_at') or a.get('created_at')
        pub_date_str = (
            pub_date.strftime('%a, %d %b %Y %H:%M:%S +0300')
            if hasattr(pub_date, 'strftime') else str(pub_date or '')
        ) or datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0300')
        guid = str(a.get('id') or slug)
        items.append(f"""    <item>
      <title>{html.escape(title)}</title>
      <link>{article_url}</link>
      <guid>{guid}</guid>
      <pubDate>{pub_date_str}</pubDate>
      <author>{html.escape(author)}</author>
      <description>{html.escape(teaser[:500])}</description>
      {enclosure}
      <category>format-article</category>
      <content:encoded><![CDATA[{content_html}]]></content:encoded>
    </item>""")
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
  xmlns:content="http://purl.org/rss/1.0/modules/content/"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:media="http://search.yahoo.com/mrss/"
  xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>{SITE_TITLE}</title>
    <link>{SITE_URL}</link>
    <description>{SITE_DESCRIPTION}</description>
    <language>{SITE_LANGUAGE}</language>
    <atom:link href="{SITE_URL}/feed/dzen.xml" rel="self" type="application/rss+xml"/>
{"".join(items)}
  </channel>
</rss>"""
    return Response(content=rss, media_type="application/rss+xml; charset=utf-8")


@app.get("/feed/rss.xml")
async def standard_rss():
    articles = get_articles(limit=50)
    items = []
    for a in articles:
        title       = a['display_title']
        article_url = f"{SITE_URL}/article/{a['slug']}"
        teaser      = a.get('teaser') or ''
        pub_date    = a.get('updated_at') or a.get('created_at') or datetime.now()
        pub_date_str = (
            pub_date.strftime('%a, %d %b %Y %H:%M:%S +0300')
            if hasattr(pub_date, 'strftime') else str(pub_date)
        )
        items.append(f"""    <item>
      <title>{html.escape(title)}</title>
      <link>{article_url}</link>
      <guid>{str(a.get('id', a['slug']))}</guid>
      <pubDate>{pub_date_str}</pubDate>
      <description>{html.escape(teaser[:500])}</description>
    </item>""")
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>{SITE_TITLE}</title>
    <link>{SITE_URL}</link>
    <description>{SITE_DESCRIPTION}</description>
    <language>{SITE_LANGUAGE}</language>
{"".join(items)}
  </channel>
</rss>"""
    return Response(content=rss, media_type="application/rss+xml; charset=utf-8")


# =================================================================
# robots.txt
# =================================================================

@app.get("/robots.txt")
async def robots():
    return PlainTextResponse(f"""User-agent: *
Allow: /
Allow: /feed/dzen.xml
Allow: /article/

Sitemap: {SITE_URL}/sitemap.xml
""")


# =================================================================
# JSON API
# =================================================================

@app.get("/api/articles")
async def api_articles(status: str = "published", limit: int = 20):
    articles = get_articles(status=status, limit=limit)
    return {"count": len(articles), "articles": articles}


# =================================================================
# Запуск
# =================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=False)