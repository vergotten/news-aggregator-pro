# -*- coding: utf-8 -*-
"""
web/app.py — FastAPI блог + RSS для Дзена

Логика модерации:
  - Сайт показывает все published статьи
  - Дзен фид — только published + dzen_approved=TRUE
  - Флаг dzen_approved включается в админке вручную
"""

import os, re, html, sys
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import psycopg2, psycopg2.extras

# .env loading
for _p in [os.path.join(os.path.dirname(__file__), ".env"),
           os.path.join(os.path.dirname(__file__), "..", ".env")]:
    if os.path.exists(_p):
        with open(_p) as _f:
            for _l in _f:
                _l = _l.strip()
                if _l and not _l.startswith("#") and "=" in _l:
                    k, _, v = _l.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        break

SITE_TITLE       = "НейроКотΔ"
SITE_DESCRIPTION = "Технические статьи с AI-обработкой"
SITE_URL         = os.getenv("SITE_URL", "http://localhost:8080")
SITE_LANGUAGE    = "ru"

def get_conn():
    url = os.getenv("DATABASE_URL", "")
    if not url: raise RuntimeError("DATABASE_URL не задан")
    return psycopg2.connect(url)

app = FastAPI(title=SITE_TITLE, docs_url="/api/docs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from admin_routes import router as admin_router
app.include_router(admin_router)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

def make_slug(title: str) -> str:
    tr = {'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'yo','ж':'zh','з':'z',
          'и':'i','й':'j','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r',
          'с':'s','т':'t','у':'u','ф':'f','х':'h','ц':'ts','ч':'ch','ш':'sh',
          'щ':'sch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya'}
    r = ""
    for c in title.lower().strip():
        r += tr.get(c, c if c.isalnum() else '-' if c in ' -_' else '')
    return re.sub(r'-+', '-', r).strip('-')[:80]

def markdown_to_html(text: str) -> str:
    if not text: return ""
    text = re.sub(r'```(\w*)\n(.*?)```', r'<pre><code>\2</code></pre>', text, flags=re.DOTALL)
    text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$',  r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',   r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    lines, in_list, result = text.split('\n'), False, []
    for line in lines:
        s = line.strip()
        if re.match(r'^[-*•]\s+', s):
            if not in_list: result.append('<ul>'); in_list = True
            item = re.sub(r'^[-*•]\s+', '', s)
            result.append(f'<li>{item}</li>')
        else:
            if in_list: result.append('</ul>'); in_list = False
            result.append(line)
    if in_list: result.append('</ul>')
    text = '\n'.join(result)
    processed = []
    for p in text.split('\n\n'):
        p = p.strip()
        if not p: continue
        if p.startswith(('<h','<pre','<ul','<ol','<blockquote','<figure')):
            processed.append(p)
        else:
            processed.append(f'<p>{p.replace(chr(10), "<br>")}</p>')
    return '\n'.join(processed)

def get_articles(status="published", limit=20, dzen_only=False) -> List[dict]:
    c = get_conn()
    cur = c.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    dzen_filter = "AND dzen_approved = TRUE" if dzen_only else ""
    cur.execute(f"""
        SELECT id, title, editorial_title, editorial_teaser, editorial_rewritten,
               telegraph_content_html, telegram_cover_image, telegraph_url,
               tags, images, relevance_score, author, source, url,
               published_at, created_at, updated_at,
               COALESCE(dzen_approved, FALSE) as dzen_approved
        FROM articles
        WHERE status = %s AND relevance_score >= 5 {dzen_filter}
        ORDER BY updated_at DESC LIMIT %s
    """, (status, limit))
    rows = cur.fetchall(); cur.close(); c.close()
    articles = []
    for r in rows:
        r = dict(r)
        title = r.get('editorial_title') or r.get('title') or 'Без заголовка'
        r['display_title'] = title
        r['slug'] = make_slug(title)
        r['teaser'] = r.get('editorial_teaser') or ''
        r['cover_image'] = r.get('telegram_cover_image') or ((r.get('images') or [''])[0] if r.get('images') else '')
        articles.append(r)
    return articles

def get_article_by_slug(slug):
    for a in get_articles(limit=100):
        if a['slug'] == slug: return a
    return None

def _is_admin(request: Request) -> bool:
    try:
        from admin_routes import verify_session_token
        token = request.cookies.get("admin_session", "")
        return bool(token) and verify_session_token(token)
    except: return False

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try: articles = get_articles(limit=20)
    except: articles = []
    return templates.TemplateResponse("index.html", {
        "request": request, "articles": articles,
        "site_title": SITE_TITLE, "site_url": SITE_URL, "is_admin": _is_admin(request)})

@app.get("/article/{slug}", response_class=HTMLResponse)
async def article_page(request: Request, slug: str):
    article = get_article_by_slug(slug)
    if not article: raise HTTPException(status_code=404, detail="Статья не найдена")
    content_html = markdown_to_html(article.get('telegraph_content_html') or article.get('editorial_rewritten') or '')
    return templates.TemplateResponse("article.html", {
        "request": request, "article": article, "content_html": content_html,
        "site_title": SITE_TITLE, "site_url": SITE_URL, "is_admin": _is_admin(request)})

@app.get("/feed/dzen.xml")
async def dzen_rss():
    """RSS для Дзена — только статьи с dzen_approved=TRUE."""
    articles = get_articles(limit=50, dzen_only=True)
    items = []
    for a in articles:
        title       = a['display_title']
        article_url = f"{SITE_URL}/article/{a['slug']}"
        teaser      = a.get('teaser') or ''
        author      = a.get('author') or ''
        content_html = markdown_to_html(a.get('telegraph_content_html') or a.get('editorial_rewritten') or '')
        cover = a.get('cover_image') or ''
        enclosure = f'<enclosure url="{html.escape(cover)}" type="image/jpeg"/>' if cover else ''
        pub_date = a.get('published_at') or a.get('updated_at') or a.get('created_at')
        pub_date_str = (pub_date.strftime('%a, %d %b %Y %H:%M:%S +0300')
                        if hasattr(pub_date, 'strftime')
                        else datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0300'))
        author_tag = f'<author>{html.escape(author)}</author>' if author else ''
        items.append(f"""    <item>
      <title>{html.escape(title)}</title>
      <link>{article_url}</link>
      <guid isPermaLink="true">{article_url}</guid>
      <pubDate>{pub_date_str}</pubDate>
      {author_tag}
      <description>{html.escape(teaser[:500])}</description>
      {enclosure}
      <category>format-article</category>
      <category>index</category>
      <category>comment-all</category>
      <content:encoded><![CDATA[<h2>{html.escape(title)}</h2>
{content_html}]]></content:encoded>
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
        title = a['display_title']
        article_url = f"{SITE_URL}/article/{a['slug']}"
        teaser = a.get('teaser') or ''
        pub_date = a.get('updated_at') or a.get('created_at') or datetime.now()
        pub_date_str = (pub_date.strftime('%a, %d %b %Y %H:%M:%S +0300')
                        if hasattr(pub_date, 'strftime') else str(pub_date))
        items.append(f"""    <item>
      <title>{html.escape(title)}</title>
      <link>{article_url}</link>
      <guid isPermaLink="true">{article_url}</guid>
      <pubDate>{pub_date_str}</pubDate>
      <description>{html.escape(teaser[:500])}</description>
    </item>""")
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>{SITE_TITLE}</title><link>{SITE_URL}</link>
    <description>{SITE_DESCRIPTION}</description>
    <language>{SITE_LANGUAGE}</language>
    <atom:link href="{SITE_URL}/feed/rss.xml" rel="self" type="application/rss+xml"/>
{"".join(items)}
  </channel>
</rss>"""
    return Response(content=rss, media_type="application/rss+xml; charset=utf-8")

@app.get("/robots.txt")
async def robots():
    return PlainTextResponse(f"User-agent: *\nAllow: /\nAllow: /feed/dzen.xml\nAllow: /feed/rss.xml\nAllow: /article/\n\nSitemap: {SITE_URL}/sitemap.xml\n")

@app.get("/api/articles")
async def api_articles(status: str = "published", limit: int = 20):
    return {"count": 0, "articles": get_articles(status=status, limit=limit)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=False)