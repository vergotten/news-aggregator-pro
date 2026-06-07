# -*- coding: utf-8 -*-
"""
News Aggregator — Streamlit Dashboard & Editor v3.

- Click row in Статьи → opens in Редактор
- Typeahead search by title in Редактор
- Original fields read-only, AI/editorial editable
"""

import os
import json
import html
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, text, inspect

# =============================================================================
# Импорт страницы настроек (settings_page.py в той же папке)
# =============================================================================

import sys
from pathlib import Path

# Добавляем текущую директорию в путь поиска модулей
_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

try:
    from settings_page import render_settings_page
except ImportError as e:
    # Если settings_page.py не найден — заглушка
    def render_settings_page():
        st.error(f"⚠️ Не удалось загрузить settings_page.py: {e}")
        st.info(f"Ожидаемый путь: {_app_dir / 'settings_page.py'}")

# Корень проекта (на уровень выше от streamlit_app/)
PROJECT_ROOT = _app_dir.parent
MODELS_CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"

# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="News Aggregator",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 8px;
        padding: 12px 16px; border-left: 4px solid #4e8cff;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; border-radius: 8px 8px 0 0;
    }
    .readonly-label {
        font-size: 0.85em; font-weight: 600;
        color: #555; margin-bottom: 2px;
    }
    .readonly-value {
        background: #f1f3f5; border: 1px solid #dee2e6;
        border-radius: 6px; padding: 8px 12px;
        font-size: 0.92em; color: #333;
        margin-bottom: 12px; white-space: pre-wrap;
        word-break: break-word; max-height: 400px; overflow-y: auto;
    }
    .readonly-tag {
        display: inline-block; background: #e9ecef;
        border-radius: 12px; padding: 2px 10px;
        font-size: 0.82em; margin: 2px 4px 2px 0;
    }
    /* Compact article list */
    .article-row hr {
        margin: 2px 0; border: none;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DB
# =============================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://newsaggregator:changeme123@postgres:5432/news_aggregator",
)

STATUS_OPTIONS = [
    "pending", "processing", "processed",
    "failed", "duplicate", "rejected", "archived",
]
SOURCE_OPTIONS = ["habr", "telegram", "reddit", "medium", "rss", "custom"]
EMBEDDING_OPTIONS = ["pending", "completed", "failed"]


@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5)


def check_db_ready() -> bool:
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "articles" in inspect(eng).get_table_names()
    except Exception as e:
        st.error(f"❌ БД недоступна: `{e}`")
        return False


def split_csv(val: str) -> List[str]:
    return [t.strip() for t in val.split(",") if t.strip()] if val else []


def esc(val: Any) -> str:
    """HTML-escape a value for safe rendering."""
    return html.escape(str(val)) if val else "—"


# =============================================================================
# Read-only display helpers
# =============================================================================

def ro_field(label: str, value: Any) -> None:
    st.markdown(
        f'<div class="readonly-label">{esc(label)}</div>'
        f'<div class="readonly-value">{esc(value)}</div>',
        unsafe_allow_html=True,
    )


def ro_tags(label: str, items: Optional[List[str]]) -> None:
    st.markdown(f'<div class="readonly-label">{esc(label)}</div>', unsafe_allow_html=True)
    if items:
        tags_html = "".join(f'<span class="readonly-tag">{esc(t)}</span>' for t in items)
        st.markdown(tags_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="readonly-value">—</div>', unsafe_allow_html=True)


# =============================================================================
# Data access
# =============================================================================

@st.cache_data(ttl=30)
def load_articles_df(
    status_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
    is_news_filter: Optional[bool] = None,
    search_query: str = "",
    min_score: float = 0.0,
    max_score: float = 10.0,
    limit: int = 500,
) -> pd.DataFrame:
    clauses = []
    params = {}
    if status_filter:
        clauses.append("status = :status")
        params["status"] = status_filter
    if source_filter:
        clauses.append("source = :source")
        params["source"] = source_filter
    if is_news_filter is not None:
        clauses.append("is_news = :is_news")
        params["is_news"] = is_news_filter
    if search_query:
        clauses.append("(title ILIKE :q OR editorial_title ILIKE :q)")
        params["q"] = f"%{search_query}%"
    clauses.append(
        "(relevance_score IS NULL OR relevance_score BETWEEN :min_s AND :max_s)"
    )
    params["min_s"] = min_score
    params["max_s"] = max_score
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"""
        SELECT id, title, editorial_title, url, source, author,
               status, is_news, relevance_score, relevance_reason,
               tags, hubs, published_at, created_at, updated_at,
               embedding_status, seo_slug
        FROM articles {where}
        ORDER BY created_at DESC LIMIT :lim
    """
    params["lim"] = limit
    try:
        with get_engine().connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_article_titles_index() -> pd.DataFrame:
    """Load id+title for typeahead search. Lightweight query."""
    try:
        with get_engine().connect() as conn:
            return pd.read_sql(
                text("""
                    SELECT id::text, title, editorial_title,
                           source, relevance_score, created_at
                    FROM articles
                    ORDER BY created_at DESC
                    LIMIT 2000
                """),
                conn,
            )
    except Exception:
        return pd.DataFrame()


def load_article_full(article_id: str) -> Optional[Dict[str, Any]]:
    try:
        with get_engine().connect() as conn:
            row = conn.execute(
                text("SELECT * FROM articles WHERE id = :aid"),
                {"aid": article_id},
            ).mappings().first()
        return dict(row) if row else None
    except Exception as e:
        st.error(f"Ошибка загрузки статьи: {e}")
        return None


def save_article(article_id: str, updates: Dict[str, Any]) -> bool:
    updates["updated_at"] = datetime.now(timezone.utc)
    set_clauses = ", ".join(f"{k} = :{k}" for k in updates)
    sql = f"UPDATE articles SET {set_clauses} WHERE id = :aid"
    updates["aid"] = article_id
    try:
        with get_engine().begin() as conn:
            conn.execute(text(sql), updates)
        return True
    except Exception as e:
        st.error(f"Ошибка сохранения: {e}")
        return False


def delete_article(article_id: str) -> bool:
    try:
        with get_engine().begin() as conn:
            conn.execute(text("DELETE FROM articles WHERE id = :aid"), {"aid": article_id})
        return True
    except Exception as e:
        st.error(f"Ошибка удаления: {e}")
        return False


@st.cache_data(ttl=60)
def load_stats() -> Dict[str, Any]:
    try:
        with get_engine().connect() as conn:
            total = conn.execute(text("SELECT count(*) FROM articles")).scalar() or 0
            by_status = dict(conn.execute(text(
                "SELECT status, count(*) FROM articles GROUP BY status"
            )).fetchall())
            by_source = dict(conn.execute(text(
                "SELECT source, count(*) FROM articles GROUP BY source"
            )).fetchall())
            avg_score = conn.execute(text(
                "SELECT avg(relevance_score) FROM articles WHERE relevance_score IS NOT NULL"
            )).scalar()
            news_count = conn.execute(text(
                "SELECT count(*) FROM articles WHERE is_news = true"
            )).scalar() or 0
            score_dist = pd.read_sql(text("""
                SELECT CASE
                    WHEN relevance_score IS NULL THEN 'N/A'
                    WHEN relevance_score < 3 THEN '0-2'
                    WHEN relevance_score < 5 THEN '3-4'
                    WHEN relevance_score < 7 THEN '5-6'
                    WHEN relevance_score < 9 THEN '7-8'
                    ELSE '9-10'
                END AS bucket, count(*) AS cnt
                FROM articles GROUP BY bucket ORDER BY bucket
            """), conn)
            daily = pd.read_sql(text("""
                SELECT date_trunc('day', created_at)::date AS day, count(*) AS cnt
                FROM articles GROUP BY day ORDER BY day DESC LIMIT 30
            """), conn)
        return {
            "total": total, "by_status": by_status, "by_source": by_source,
            "avg_score": round(avg_score, 2) if avg_score else 0,
            "news_count": news_count, "score_dist": score_dist, "daily": daily,
        }
    except Exception as e:
        st.error(f"Ошибка статистики: {e}")
        return {
            "total": 0, "by_status": {}, "by_source": {},
            "avg_score": 0, "news_count": 0,
            "score_dist": pd.DataFrame(), "daily": pd.DataFrame(),
        }


# =============================================================================
# Navigation
# =============================================================================

PAGES = ["📊 Дашборд", "📋 Статьи", "✏️ Редактор", "⚙️ Настройки"]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "📋 Статьи"

st.sidebar.title("📰 News Aggregator")
page = st.sidebar.radio(
    "Навигация", PAGES,
    index=PAGES.index(st.session_state["nav_page"]),
    key="nav_radio",
)
st.session_state["nav_page"] = page

if not check_db_ready():
    st.warning("⚠️ БД недоступна или таблица `articles` не найдена.")
    st.stop()


# =============================================================================
# PAGE: Dashboard
# =============================================================================

if page == "📊 Дашборд":
    st.title("📊 Аналитика")
    stats = load_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего статей", stats["total"])
    c2.metric("Новостей", stats["news_count"])
    c3.metric("Средний score", stats["avg_score"])
    c4.metric("Обработано", stats["by_status"].get("processed", 0))

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("По статусам")
        if stats["by_status"]:
            sdf = pd.DataFrame(list(stats["by_status"].items()), columns=["Статус", "Кол-во"])
            cmap = {
                "processed": "#28a745", "pending": "#ffc107", "failed": "#dc3545",
                "processing": "#17a2b8", "duplicate": "#6c757d",
                "rejected": "#e83e8c", "archived": "#6610f2",
            }
            fig = px.pie(sdf, names="Статус", values="Кол-во", color="Статус",
                         color_discrete_map=cmap, hole=0.4)
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных")

    with col_r:
        st.subheader("По источникам")
        if stats["by_source"]:
            srcdf = pd.DataFrame(list(stats["by_source"].items()), columns=["Источник", "Кол-во"])
            fig2 = px.bar(srcdf, x="Источник", y="Кол-во", color="Источник", text_auto=True)
            fig2.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Нет данных")

    st.subheader("Распределение Relevance Score")
    if not stats["score_dist"].empty:
        fig3 = px.bar(stats["score_dist"], x="bucket", y="cnt",
                       labels={"bucket": "Score", "cnt": "Кол-во"},
                       color="cnt", color_continuous_scale="Viridis", text_auto=True)
        fig3.update_layout(margin=dict(t=20, b=40), height=300)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Статьи по дням (30 дней)")
    if not stats["daily"].empty:
        fig4 = px.area(stats["daily"].sort_values("day"), x="day", y="cnt",
                        labels={"day": "Дата", "cnt": "Статей"})
        fig4.update_layout(margin=dict(t=20, b=40), height=280)
        st.plotly_chart(fig4, use_container_width=True)


# =============================================================================
# PAGE: Articles list  — click row → editor
# =============================================================================

elif page == "📋 Статьи":
    st.title("📋 Список статей")

    # ---- Filters ----
    with st.expander("🔍 Фильтры", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            f_status = st.selectbox("Статус", ["Все"] + STATUS_OPTIONS)
        with fc2:
            f_source = st.selectbox("Источник", ["Все"] + SOURCE_OPTIONS)
        with fc3:
            f_news = st.selectbox("Только новости?", ["Все", "Да", "Нет"])
        with fc4:
            f_search = st.text_input("Поиск по заголовку")
        f_min, f_max = st.slider("Relevance score", 0.0, 10.0, (0.0, 10.0), 0.5)

    df = load_articles_df(
        status_filter=f_status if f_status != "Все" else None,
        source_filter=f_source if f_source != "Все" else None,
        is_news_filter={"Да": True, "Нет": False}.get(f_news),
        search_query=f_search,
        min_score=f_min, max_score=f_max,
    )

    st.caption(f"Найдено: **{len(df)}** статей")

    if df.empty:
        st.info("Нет статей по заданным фильтрам.")
    else:
        # Prepare display columns
        display_df = df.copy()
        display_df["score"] = display_df["relevance_score"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "—"
        )
        display_df["display_title"] = display_df.apply(
            lambda r: r["editorial_title"]
            if pd.notna(r.get("editorial_title")) else r["title"],
            axis=1,
        ).str[:90]

        # Render each article as a clickable card-row
        for i, (_, row) in enumerate(df.iterrows()):
            title = display_df.iloc[i]["display_title"]
            score = display_df.iloc[i]["score"]
            status = row["status"]
            source = row["source"]
            created = str(row.get("created_at", ""))[:16]
            full_id = str(row["id"])
            is_news = "📰" if row.get("is_news") else ""

            col_title, col_meta, col_btn = st.columns([5, 3, 1])
            with col_title:
                st.markdown(
                    f"**{title}** {is_news}",
                    help=full_id,
                )
            with col_meta:
                st.caption(f"{source} · `{status}` · score **{score}** · {created}")
            with col_btn:
                if st.button("✏️", key=f"open_{i}", help="Открыть в редакторе"):
                    st.session_state["edit_article_id"] = full_id
                    st.session_state["nav_page"] = "✏️ Редактор"
                    st.rerun()

            if i < len(df) - 1:
                st.markdown(
                    '<hr style="margin:2px 0;border:none;border-top:1px solid #eee">',
                    unsafe_allow_html=True,
                )


# =============================================================================
# PAGE: Editor — with typeahead search
# =============================================================================

elif page == "✏️ Редактор":
    st.title("✏️ Редактор статьи")

    # ---- Typeahead search ----
    st.subheader("🔍 Найти статью")

    titles_df = load_article_titles_index()

    if titles_df.empty:
        st.warning("В БД нет статей.")
        st.stop()

    # Build searchable options: "title (source | score | date)"
    search_options = []
    id_map = {}  # label → full uuid
    for _, row in titles_df.iterrows():
        title = (row.get("editorial_title") or row.get("title") or "—")[:80]
        score = f"{row['relevance_score']:.1f}" if pd.notna(row.get("relevance_score")) else "—"
        src = row.get("source", "")
        date = str(row.get("created_at", ""))[:10]
        label = f"{title}  ·  {src}  ·  score {score}  ·  {date}"
        search_options.append(label)
        id_map[label] = str(row["id"])

    # Pre-select if coming from Статьи tab
    preselected_id = st.session_state.get("edit_article_id", "")
    default_index = 0
    if preselected_id:
        for i, label in enumerate(search_options):
            if id_map[label] == preselected_id:
                default_index = i + 1  # +1 because we prepend ""
                break

    chosen = st.selectbox(
        "Начните вводить название статьи…",
        options=[""] + search_options,
        index=default_index,
        placeholder="Поиск по заголовку…",
        key="article_search",
    )

    if not chosen:
        st.info("👆 Начните вводить название — появятся подсказки.")
        st.stop()

    edit_id = id_map.get(chosen, "")
    if not edit_id:
        st.error("Не удалось определить ID статьи.")
        st.stop()

    # Store for persistence
    st.session_state["edit_article_id"] = edit_id

    article = load_article_full(edit_id)
    if not article:
        st.error(f"Статья `{edit_id}` не найдена.")
        st.stop()

    # ---- Header ----
    st.divider()
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.subheader(article.get("title", "—"))
        if article.get("url"):
            st.markdown(f"[🔗 Открыть оригинал]({article['url']})")
    with col_h2:
        score_val = article.get("relevance_score")
        st.caption(f"Score: **{score_val:.1f}**/10" if score_val else "Score: —")
        st.caption(f"Статус: `{article.get('status', '—')}`")
        st.caption(f"Создана: {article.get('created_at', '—')}")

    st.divider()

    # ==================================================================
    # TABS
    # ==================================================================
    tab_orig, tab_ai, tab_tg, tab_seo, tab_meta = st.tabs([
        "📄 Оригинал (только чтение)",
        "🤖 AI / Редакторский контент",
        "📱 Telegram",
        "🔍 SEO",
        "⚙️ Метаданные",
    ])

    # ==== Original — READ ONLY ====
    with tab_orig:
        st.caption("🔒 Оригинальные данные из парсера. Не редактируются.")
        c1, c2 = st.columns(2)
        with c1:
            ro_field("Заголовок", article.get("title"))
            ro_field("URL", article.get("url"))
            ro_field("Автор", article.get("author"))
            ro_field("Источник", article.get("source"))
        with c2:
            ro_field("Дата публикации", article.get("published_at"))
            ro_field("Создана в БД", article.get("created_at"))
            ro_field("Обновлена", article.get("updated_at"))
            ro_field("ID", str(article.get("id", "")))
        ro_tags("Теги", article.get("tags"))
        ro_tags("Хабы", article.get("hubs"))
        ro_field("Контент (оригинал)", article.get("content"))

    # ==== Editable form (tabs 2-5) ====
    with st.form("article_editor", clear_on_submit=False):

        with tab_ai:
            st.caption("✏️ Редактируемые поля — результаты AI обработки")
            c1, c2 = st.columns(2)
            with c1:
                ed_status = st.selectbox(
                    "Статус", STATUS_OPTIONS,
                    index=STATUS_OPTIONS.index(article["status"])
                    if article.get("status") in STATUS_OPTIONS else 0,
                )
                ed_is_news = st.checkbox("Новость", value=bool(article.get("is_news")))
                ed_score = st.number_input(
                    "Relevance Score", min_value=0.0, max_value=10.0, step=0.5,
                    value=float(article.get("relevance_score") or 0),
                )
                ed_reason = st.text_area(
                    "Relevance Reason",
                    value=article.get("relevance_reason", "") or "", height=100,
                )
            with c2:
                ed_editorial_title = st.text_input(
                    "Editorial Title",
                    value=article.get("editorial_title", "") or "",
                )
                ed_editorial_teaser = st.text_area(
                    "Editorial Teaser",
                    value=article.get("editorial_teaser", "") or "", height=100,
                )
            ed_editorial_rewritten = st.text_area(
                "Editorial Rewritten (полный переписанный текст)",
                value=article.get("editorial_rewritten", "") or "", height=300,
            )

        with tab_tg:
            st.caption("✏️ Telegram / Telegraph контент")
            ed_tg_text = st.text_area(
                "Telegram Post Text (HTML)",
                value=article.get("telegram_post_text", "") or "", height=200,
            )
            ed_tg_cover = st.text_input(
                "Cover Image URL",
                value=article.get("telegram_cover_image", "") or "",
            )
            ed_telegraph_url = st.text_input(
                "Telegraph URL",
                value=article.get("telegraph_url", "") or "",
            )
            ed_telegraph_html = st.text_area(
                "Telegraph Content HTML",
                value=article.get("telegraph_content_html", "") or "", height=200,
            )

        with tab_seo:
            st.caption("✏️ SEO оптимизация")
            c1, c2 = st.columns(2)
            with c1:
                ed_seo_title = st.text_input(
                    "SEO Title (50-60 chars)",
                    value=article.get("seo_title", "") or "",
                )
                ed_seo_slug = st.text_input(
                    "SEO Slug", value=article.get("seo_slug", "") or "",
                )
                ed_seo_focus = st.text_input(
                    "Focus Keyword",
                    value=article.get("seo_focus_keyword", "") or "",
                )
            with c2:
                ed_seo_desc = st.text_area(
                    "SEO Description (150-160 chars)",
                    value=article.get("seo_description", "") or "", height=100,
                )
                ed_seo_kw = st.text_input(
                    "SEO Keywords (через запятую)",
                    value=", ".join(article.get("seo_keywords") or []),
                )

        with tab_meta:
            st.caption("✏️ Технические метаданные")
            c1, c2 = st.columns(2)
            with c1:
                ed_embed_status = st.selectbox(
                    "Embedding Status", EMBEDDING_OPTIONS,
                    index=EMBEDDING_OPTIONS.index(article["embedding_status"])
                    if article.get("embedding_status") in EMBEDDING_OPTIONS else 0,
                )
                ed_vector_id = st.text_input(
                    "Vector ID", value=article.get("vector_id", "") or "",
                )
            with c2:
                ed_images = st.text_area(
                    "Images (по одному URL на строку)",
                    value="\n".join(article.get("images") or []), height=100,
                )
            ed_meta_json = st.text_area(
                "article_metadata (JSON)",
                value=json.dumps(
                    article.get("article_metadata") or {},
                    ensure_ascii=False, indent=2,
                ),
                height=200,
            )

        # ==== Buttons ====
        st.divider()
        bc1, _bc2, bc3 = st.columns([2, 1, 1])
        with bc1:
            submitted = st.form_submit_button(
                "💾 Сохранить изменения", type="primary", use_container_width=True
            )
        with bc3:
            delete_clicked = st.form_submit_button(
                "🗑️ Удалить статью", use_container_width=True
            )

    # ---- Save ----
    if submitted:
        try:
            meta_parsed = json.loads(ed_meta_json) if ed_meta_json.strip() else {}
        except json.JSONDecodeError:
            st.error("Невалидный JSON в article_metadata")
            st.stop()

        updates = {
            "status": ed_status,
            "is_news": ed_is_news,
            "relevance_score": ed_score if ed_score > 0 else None,
            "relevance_reason": ed_reason or None,
            "editorial_title": ed_editorial_title or None,
            "editorial_teaser": ed_editorial_teaser or None,
            "editorial_rewritten": ed_editorial_rewritten or None,
            "telegram_post_text": ed_tg_text or None,
            "telegram_cover_image": ed_tg_cover or None,
            "telegraph_url": ed_telegraph_url or None,
            "telegraph_content_html": ed_telegraph_html or None,
            "seo_title": ed_seo_title or None,
            "seo_description": ed_seo_desc or None,
            "seo_slug": ed_seo_slug or None,
            "seo_keywords": split_csv(ed_seo_kw),
            "seo_focus_keyword": ed_seo_focus or None,
            "embedding_status": ed_embed_status,
            "vector_id": ed_vector_id or None,
            "images": [u.strip() for u in ed_images.strip().splitlines() if u.strip()],
            "article_metadata": meta_parsed,
        }

        if save_article(edit_id, updates):
            st.success("✅ Сохранено!")
            load_articles_df.clear()
            load_article_titles_index.clear()
            load_stats.clear()

    # ---- Delete ----
    if delete_clicked:
        if delete_article(edit_id):
            st.warning("🗑️ Статья удалена.")
            st.session_state.pop("edit_article_id", None)
            load_articles_df.clear()
            load_article_titles_index.clear()
            load_stats.clear()

# =============================================================================
# PAGE: Settings
# =============================================================================

elif page == "⚙️ Настройки":
    render_settings_page()