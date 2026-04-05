# -*- coding: utf-8 -*-
"""
web/admin_routes.py — Admin panel routes

Routes:
  GET  /admin/login       — login page
  POST /admin/login       — authenticate
  GET  /admin/logout      — logout
  GET  /admin             — article list (with filters)
  GET  /admin/article/{id} — edit form
  POST /admin/article/{id} — save changes
"""

import os
import hmac
import hashlib
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import psycopg2
import psycopg2.extras

router = APIRouter(prefix="/admin")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# ─────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-in-prod")
SESSION_COOKIE = "admin_session"


def _sign(value: str) -> str:
    return hmac.new(SESSION_SECRET.encode(), value.encode(), hashlib.sha256).hexdigest()


def make_session_token(username: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H")  # valid for 1 hour
    payload = f"{username}:{ts}"
    return f"{payload}:{_sign(payload)}"


def verify_session_token(token: str) -> bool:
    try:
        parts = token.rsplit(":", 1)
        if len(parts) != 2:
            return False
        payload, sig = parts
        expected = _sign(payload)
        return hmac.compare_digest(expected, sig)
    except Exception:
        return False


def is_authenticated(request: Request) -> bool:
    token = request.cookies.get(SESSION_COOKIE, "")
    return bool(token) and verify_session_token(token)


def require_auth(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=302, headers={"Location": "/admin/login"})


# ─────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────

def get_conn():
    url = os.getenv("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(url)


def fetch_articles(status: Optional[str] = None, search: Optional[str] = None, limit: int = 50):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    conditions = []
    params = []

    if status and status != "all":
        conditions.append("status = %s")
        params.append(status)
    if search:
        conditions.append("(title ILIKE %s OR editorial_title ILIKE %s)")
        params.extend([f"%{search}%", f"%{search}%"])

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    cur.execute(f"""
        SELECT id, title, editorial_title, editorial_teaser, status,
               relevance_score, tags, source, created_at, updated_at, telegraph_url
        FROM articles
        {where}
        ORDER BY updated_at DESC
        LIMIT %s
    """, params)

    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def fetch_article(article_id: str):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT *, COALESCE(dzen_approved, FALSE) as dzen_approved FROM articles WHERE id = %s", (article_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else None


def update_article(article_id: str, data: dict):
    if not data:
        return
    conn = get_conn()
    cur = conn.cursor()

    allowed = {
        "editorial_title", "editorial_teaser", "editorial_rewritten",
        "telegram_post_text", "telegraph_content_html", "relevance_score",
        "status", "tags", "title", "author", "source", "url",
        "telegram_cover_image", "dzen_approved", "cat_comment", "cat_comment",
    }
    filtered = {k: v for k, v in data.items() if k in allowed and v is not None}
    filtered["updated_at"] = datetime.utcnow()

    set_clause = ", ".join(f"{k} = %s" for k in filtered)
    values = list(filtered.values()) + [article_id]

    cur.execute(f"UPDATE articles SET {set_clause} WHERE id = %s", values)
    conn.commit()
    cur.close()
    conn.close()


def get_stats():
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status = 'pending') as pending,
            COUNT(*) FILTER (WHERE status = 'processed') as processed,
            COUNT(*) FILTER (WHERE status = 'published') as published,
            ROUND(AVG(relevance_score)::numeric, 1) as avg_score
        FROM articles
    """)
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else {}


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    return templates.TemplateResponse("admin_login.html", {
        "request": request,
        "error": error,
    })


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    if username == ADMIN_USER and password == ADMIN_PASSWORD:
        token = make_session_token(username)
        response = RedirectResponse(url="/admin", status_code=303)
        response.set_cookie(
            SESSION_COOKIE, token,
            httponly=True, samesite="lax",
            max_age=60 * 60 * 8,  # 8 hours
        )
        return response
    return templates.TemplateResponse("admin_login.html", {
        "request": request,
        "error": "Неверный логин или пароль",
    }, status_code=401)


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


@router.get("", response_class=HTMLResponse)
async def admin_index(
    request: Request,
    status: str = "all",
    search: str = "",
    limit: int = 50,
):
    if not is_authenticated(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    try:
        articles = fetch_articles(
            status=status if status != "all" else None,
            search=search or None,
            limit=limit,
        )
        stats = get_stats()
        error = None
    except Exception as e:
        articles = []
        stats = {}
        error = str(e)

    return templates.TemplateResponse("admin_index.html", {
        "request": request,
        "articles": articles,
        "stats": stats,
        "current_status": status,
        "search": search,
        "error": error,
    })


@router.get("/article/{article_id}", response_class=HTMLResponse)
async def article_edit_page(request: Request, article_id: str):
    if not is_authenticated(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    article = fetch_article(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Статья не найдена")

    # Normalise tags for display
    tags = article.get("tags") or []
    if isinstance(tags, list):
        article["tags_str"] = ", ".join(tags)
    else:
        article["tags_str"] = str(tags)

    return templates.TemplateResponse("admin_article.html", {
        "request": request,
        "article": article,
        "saved": request.query_params.get("saved") == "1",
    })


@router.post("/article/{article_id}")
async def article_save(
    request: Request,
    article_id: str,
    title: str = Form(default=""),
    editorial_title: str = Form(default=""),
    editorial_teaser: str = Form(default=""),
    editorial_rewritten: str = Form(default=""),
    telegram_post_text: str = Form(default=""),
    telegraph_content_html: str = Form(default=""),
    relevance_score: str = Form(default=""),
    status: str = Form(default=""),
    tags_str: str = Form(default=""),
    author: str = Form(default=""),
    source: str = Form(default=""),
    url: str = Form(default=""),
    telegram_cover_image: str = Form(default=""),
    dzen_approved: str = Form(default=""),
    cat_comment: str = Form(default=""),
):
    if not is_authenticated(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    # Parse tags from comma-separated string
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None

    # Parse score
    score = None
    if relevance_score:
        try:
            score = float(relevance_score)
        except ValueError:
            pass

    data = {
        "title": title or None,
        "editorial_title": editorial_title or None,
        "editorial_teaser": editorial_teaser or None,
        "editorial_rewritten": editorial_rewritten or None,
        "telegram_post_text": telegram_post_text or None,
        "telegraph_content_html": telegraph_content_html or None,
        "relevance_score": score,
        "status": status or None,
        "tags": tags,
        "author": author or None,
        "source": source or None,
        "url": url or None,
        "telegram_cover_image": telegram_cover_image or None,
        "dzen_approved": dzen_approved == "on",
        "cat_comment": cat_comment or None,
    }

    update_article(article_id, data)
    return RedirectResponse(url=f"/admin/article/{article_id}?saved=1", status_code=303)


# =================================================================
# Telegraph sync endpoint
# =================================================================

TELEGRAPH_API = "https://api.telegra.ph"


def telegraph_edit_page(token: str, path: str, title: str, content_nodes: list) -> dict:
    """Обновить существующую страницу на Telegraph через editPage API."""
    import requests as _requests
    import json as _json
    resp = _requests.post(f"{TELEGRAPH_API}/editPage/{path}", data={
        "access_token": token,
        "title": title[:256],
        "content": _json.dumps(content_nodes, ensure_ascii=False),
        "return_content": "false",
    })
    resp.raise_for_status()
    return resp.json()


def content_to_nodes_simple(content: str) -> list:
    """Простая конвертация текста в Telegraph nodes для editPage."""
    import re as _re
    if not content:
        return [{"tag": "p", "children": ["Контент отсутствует"]}]
    nodes = []
    for para in content.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        # Заголовки markdown
        m = _re.match(r'^#{1,4}\s+(.+)$', para)
        if m:
            nodes.append({"tag": "h4", "children": [m.group(1)]})
            continue
        # Короткая строка без точки → подзаголовок
        if len(para) < 80 and "\n" not in para and not para.endswith("."):
            nodes.append({"tag": "h4", "children": [para]})
            continue
        nodes.append({"tag": "p", "children": [para.replace("\n", " ")]})
    return nodes or [{"tag": "p", "children": [content[:500]]}]


@router.post("/article/{article_id}/sync-telegraph")
async def sync_telegraph(request: Request, article_id: str):
    """Обновить Telegraph страницу из текущего контента в БД."""
    if not is_authenticated(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    article = fetch_article(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Статья не найдена")

    # Читаем токен из article_metadata
    metadata = article.get("article_metadata") or {}
    if isinstance(metadata, str):
        try:
            import json as _json
            metadata = _json.loads(metadata)
        except Exception:
            metadata = {}

    token = metadata.get("telegraph_access_token")
    if not token:
        raise HTTPException(
            status_code=400,
            detail="Telegraph токен не найден в metadata. Статья должна быть опубликована через publish_pending.py хотя бы один раз после этого обновления."
        )

    # Достаём path из telegraph_url (https://telegra.ph/slug-01-01 → slug-01-01)
    telegraph_url = article.get("telegraph_url") or ""
    if not telegraph_url:
        raise HTTPException(status_code=400, detail="telegraph_url не задан")

    path = telegraph_url.rstrip("/").split("/")[-1]

    # Контент
    content = article.get("telegraph_content_html") or article.get("editorial_rewritten") or ""
    title = article.get("editorial_title") or article.get("title") or "Без заголовка"
    nodes = content_to_nodes_simple(content)

    result = telegraph_edit_page(token, path, title, nodes)

    if result.get("ok"):
        return {"ok": True, "url": result["result"].get("url")}
    else:
        raise HTTPException(status_code=500, detail=f"Telegraph error: {result.get('error')}")