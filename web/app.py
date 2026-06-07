# -*- coding: utf-8 -*-
"""
web/app.py — FastAPI блог + RSS для Дзена

Логика модерации:
  - Сайт показывает все published статьи
  - Дзен фид — только published + dzen_approved=TRUE
"""

import os
import re
import sys
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
# Загрузка .env
# =================================================================

for _p in [
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(__file__), "..", ".env"),
]:
    if os.path.exists(_p):
        with open(_p) as _f:
            for _l in _f:
                _l = _l.strip()
                if _l and not _l.startswith("#") and "=" in _l:
                    k, _, v = _l.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
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
# =================================================================

def get_conn():
    url = os.getenv("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL не задан")
    return psycopg2.connect(url)

# =================================================================
# FastAPI App
# =================================================================

app = FastAPI(title=SITE_TITLE, docs_url="/api/docs")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from admin_routes import router as admin_router  # noqa: E402
app.include_router(admin_router)

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
    tr = {
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'yo',
        'ж':'zh','з':'z','и':'i','й':'j','к':'k','л':'l','м':'m',
        'н':'n','о':'o','п':'p','р':'r','с':'s','т':'t','у':'u',
        'ф':'f','х':'h','ц':'ts','ч':'ch','ш':'sh','щ':'sch',
        'ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya',
    }
    result = ""
    for c in title.lower().strip():
        if c in tr:
            result += tr[c]
        elif c.isalnum():
            result += c
        elif c in (' ', '-', '_'):
            result += '-'
    return re.sub(r'-+', '-', result).strip('-')[:80]


def markdown_to_html(text: str) -> str:
    """Конвертация markdown -> HTML для сайта (полный набор тегов)."""
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
        s = line.strip()
        if re.match(r'^[-*•]\s+', s):
            if not in_list:
                result.append('<ul>')
                in_list = True
            item = re.sub(r'^[-*•]\s+', '', s)
            result.append('<li>' + item + '</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            result.append(line)
    if in_list:
        result.append('</ul>')
    text = '\n'.join(result)
    processed = []
    for p in text.split('\n\n'):
        p = p.strip()
        if not p:
            continue
        if p.startswith(('<h', '<pre', '<ul', '<ol', '<blockquote', '<figure')):
            processed.append(p)
        else:
            processed.append('<p>' + p.replace('\n', '<br>') + '</p>')
    return '\n'.join(processed)


def markdown_to_dzen_html(text: str) -> str:
    """
    Конвертация markdown -> HTML строго по требованиям Дзена.
    Разрешённые теги: p, a, b, i, u, s, h1-h4, blockquote,
    ul/li, ol/li, figure/img/figcaption.
    НЕ поддерживаются: strong, em, code, pre, span.
    """
    if not text:
        return ""

    # Code blocks -> blockquote (Дзен не поддерживает pre/code)
    def code_block_replace(m):
        code = m.group(1).replace('\n', '<br>')
        return '<blockquote>' + code + '</blockquote>'

    text = re.sub(r'```(?:\w*)\n(.*?)```', code_block_replace, text, flags=re.DOTALL)

    # Headings
    text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$',  r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',   r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$',    r'<h1>\1</h1>', text, flags=re.MULTILINE)

    # Bold -> <b>, italic -> <i>  (не <strong>/<em> — Дзен не поддерживает)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)

    # Inline code -> plain text
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Links
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    # Images -> <figure><img>
    def img_replace(m):
        caption = m.group(1)
        src = m.group(2)
        cap_tag = '<figcaption>' + caption + '</figcaption>' if caption else ''
        return '<figure><img src="' + src + '">' + cap_tag + '</figure>'

    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', img_replace, text)

    # Blockquote
    text = re.sub(r'^>\s*(.+)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)

    # Lists (ul and ol)
    lines = text.split('\n')
    in_ul = False
    in_ol = False
    result = []
    for line in lines:
        s = line.strip()
        if re.match(r'^[-*•]\s+', s):
            if in_ol:
                result.append('</ol>')
                in_ol = False
            if not in_ul:
                result.append('<ul>')
                in_ul = True
            item = re.sub(r'^[-*•]\s+', '', s)
            result.append('<li>' + item + '</li>')
        elif re.match(r'^\d+[.)]\s+', s):
            if in_ul:
                result.append('</ul>')
                in_ul = False
            if not in_ol:
                result.append('<ol>')
                in_ol = True
            item = re.sub(r'^\d+[.)]\s+', '', s)
            result.append('<li>' + item + '</li>')
        else:
            if in_ul:
                result.append('</ul>')
                in_ul = False
            if in_ol:
                result.append('</ol>')
                in_ol = False
            result.append(line)
    if in_ul:
        result.append('</ul>')
    if in_ol:
        result.append('</ol>')
    text = '\n'.join(result)

    # Paragraphs
    processed = []
    for p in text.split('\n\n'):
        p = p.strip()
        if not p:
            continue
        if p.startswith(('<h', '<ul', '<ol', '<blockquote', '<figure')):
            processed.append(p)
        else:
            processed.append('<p>' + p.replace('\n', '<br>') + '</p>')
    return '\n'.join(processed)


def get_articles(status: str = "published", limit: int = 20,
                 dzen_only: bool = False) -> List[dict]:
    c = get_conn()
    cur = c.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    dzen_filter = "AND dzen_approved = TRUE" if dzen_only else ""
    cur.execute(
        "SELECT id, title, editorial_title, editorial_teaser, editorial_rewritten,"
        " telegraph_content_html, telegram_cover_image, telegraph_url,"
        " tags, images, relevance_score, author, source, url,"
        " published_at, created_at, updated_at,"
        " COALESCE(dzen_approved, FALSE) as dzen_approved"
        " FROM articles"
        " WHERE status = %s AND relevance_score >= 5 " + dzen_filter +
        " ORDER BY updated_at DESC LIMIT %s",
        (status, limit)
    )
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
        images = r.get('images') or []
        r['cover_image'] = r.get('telegram_cover_image') or (images[0] if images else '')
        articles.append(r)
    return articles


def get_article_by_slug(slug: str) -> Optional[dict]:
    for a in get_articles(limit=100):
        if a['slug'] == slug:
            return a
    return None


def _is_admin(request: Request) -> bool:
    try:
        from admin_routes import verify_session_token
        token = request.cookies.get("admin_session", "")
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
    content_md = (
        article.get('telegraph_content_html')
        or article.get('editorial_rewritten')
        or ''
    )
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
# RSS для Дзена — только dzen_approved статьи
# =================================================================

@app.get("/feed/dzen.xml")
async def dzen_rss():
    articles = get_articles(limit=50, dzen_only=True)
    items = []
    for a in articles:
        title       = a['display_title']
        article_url = SITE_URL + '/article/' + a['slug']
        teaser      = a.get('teaser') or ''
        author      = a.get('author') or ''
        cover       = a.get('cover_image') or ''

        content_md   = a.get('telegraph_content_html') or a.get('editorial_rewritten') or ''
        content_html = markdown_to_dzen_html(content_md)

        enclosure = ''
        if cover:
            enclosure = '<enclosure url="' + html.escape(cover) + '" type="image/jpeg"/>'

        pub_date = a.get('published_at') or a.get('updated_at') or a.get('created_at')
        if pub_date and hasattr(pub_date, 'strftime'):
            pub_date_str = pub_date.strftime('%a, %d %b %Y %H:%M:%S +0300')
        else:
            pub_date_str = datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0300')

        author_tag = ''
        if author:
            author_tag = '<author>' + html.escape(author) + '</author>'

        # Заголовок дублируем внутри content:encoded — требование Дзена
        full_content = '<h2>' + html.escape(title) + '</h2>\n' + content_html

        item = (
            '    <item>\n'
            '      <title>' + html.escape(title) + '</title>\n'
            '      <link>' + article_url + '</link>\n'
            '      <guid isPermaLink="true">' + article_url + '</guid>\n'
            '      <pubDate>' + pub_date_str + '</pubDate>\n'
            + ('      ' + author_tag + '\n' if author_tag else '')
            + '      <description>' + html.escape(teaser[:500]) + '</description>\n'
            + ('      ' + enclosure + '\n' if enclosure else '')
            + '      <category>format-article</category>\n'
            '      <category>index</category>\n'
            '      <category>comment-all</category>\n'
            '      <content:encoded><![CDATA[' + full_content + ']]></content:encoded>\n'
            '    </item>'
        )
        items.append(item)

    rss = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0"\n'
        '  xmlns:content="http://purl.org/rss/1.0/modules/content/"\n'
        '  xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        '  xmlns:media="http://search.yahoo.com/mrss/"\n'
        '  xmlns:atom="http://www.w3.org/2005/Atom">\n'
        '  <channel>\n'
        '    <title>' + SITE_TITLE + '</title>\n'
        '    <link>' + SITE_URL + '</link>\n'
        '    <description>' + SITE_DESCRIPTION + '</description>\n'
        '    <language>' + SITE_LANGUAGE + '</language>\n'
        '    <atom:link href="' + SITE_URL + '/feed/dzen.xml" rel="self" type="application/rss+xml"/>\n'
        + '\n'.join(items) + '\n'
        '  </channel>\n'
        '</rss>'
    )
    return Response(content=rss, media_type="application/rss+xml; charset=utf-8")


# =================================================================
# Обычный RSS
# =================================================================

@app.get("/feed/rss.xml")
async def standard_rss():
    articles = get_articles(limit=50)
    items = []
    for a in articles:
        title       = a['display_title']
        article_url = SITE_URL + '/article/' + a['slug']
        teaser      = a.get('teaser') or ''
        pub_date    = a.get('updated_at') or a.get('created_at') or datetime.now()
        if hasattr(pub_date, 'strftime'):
            pub_date_str = pub_date.strftime('%a, %d %b %Y %H:%M:%S +0300')
        else:
            pub_date_str = str(pub_date)
        item = (
            '    <item>\n'
            '      <title>' + html.escape(title) + '</title>\n'
            '      <link>' + article_url + '</link>\n'
            '      <guid isPermaLink="true">' + article_url + '</guid>\n'
            '      <pubDate>' + pub_date_str + '</pubDate>\n'
            '      <description>' + html.escape(teaser[:500]) + '</description>\n'
            '    </item>'
        )
        items.append(item)

    rss = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n'
        '  <channel>\n'
        '    <title>' + SITE_TITLE + '</title>\n'
        '    <link>' + SITE_URL + '</link>\n'
        '    <description>' + SITE_DESCRIPTION + '</description>\n'
        '    <language>' + SITE_LANGUAGE + '</language>\n'
        '    <atom:link href="' + SITE_URL + '/feed/rss.xml" rel="self" type="application/rss+xml"/>\n'
        + '\n'.join(items) + '\n'
        '  </channel>\n'
        '</rss>'
    )
    return Response(content=rss, media_type="application/rss+xml; charset=utf-8")


# =================================================================
# robots.txt
# =================================================================

@app.get("/robots.txt")
async def robots():
    return PlainTextResponse(
        "User-agent: *\n"
        "Allow: /\n"
        "Allow: /feed/dzen.xml\n"
        "Allow: /feed/rss.xml\n"
        "Allow: /article/\n\n"
        "Sitemap: " + SITE_URL + "/sitemap.xml\n"
    )


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