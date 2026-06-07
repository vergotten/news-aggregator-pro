# -*- coding: utf-8 -*-
"""
Страница настроек для Streamlit — ⚙️ Настройки.

Путь: streamlit_app/settings_page.py

Управляет конфигурацией LLM провайдеров, моделей и pipeline.
Настройки хранятся в config/settings.json (не требует pyyaml).
Также может читать/писать config/models.yaml если pyyaml доступен.

Все изменения в UI сохраняются в файл конфига.
Ручные правки файла отражаются в UI при перезагрузке.
"""

import os
import json
import copy
import requests
import streamlit as st
from pathlib import Path
from typing import Dict, Any, List

# PyYAML — опционально (может не быть в streamlit-контейнере)
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# =============================================================================
# Пути к конфигурации
# =============================================================================

# Корень проекта — на один уровень выше от streamlit_app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# JSON-конфиг (основной, не требует pyyaml)
SETTINGS_JSON = PROJECT_ROOT / "config" / "settings.json"

# YAML-конфиг (для совместимости с pipeline и models_config.py)
MODELS_YAML = PROJECT_ROOT / "config" / "models.yaml"

# URL сервисов
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
API_URL = os.getenv("API_URL", "http://api:8000")

# =============================================================================
# Константы
# =============================================================================

PROVIDERS = ["ollama", "groq", "openrouter", "google"]

AGENT_LABELS = {
    "classifier": "🏷️ Классификатор",
    "relevance": "📊 Релевантность",
    "quality_validator": "✅ Валидатор",
    "summarizer": "📝 Суммаризатор",
    "rewriter": "✏️ Рерайтер",
    "style_normalizer": "🎨 Нормализатор",
    "telegram_formatter": "📱 TG форматтер",
    "seo_optimizer": "🔍 SEO",
}


# =============================================================================
# Значения по умолчанию
# =============================================================================

def _defaults() -> Dict[str, Any]:
    """Дефолтная конфигурация. Используется если файл не найден."""
    return {
        "defaults": {
            "provider": "ollama",
            "strategy": "balanced",
            "enable_fallback": True,
        },
        "ollama": {
            "model": "qwen2.5:14b-instruct-q5_k_m",    # glm-4.7-flash:q4_K_M
            "base_url": "http://ollama:11434",
            "context_length": 32768,
        },
        "temperatures": {
            "classifier": 0.1,
            "relevance": 0.3,
            "quality_validator": 0.1,
            "summarizer": 0.5,
            "rewriter": 0.7,
            "style_normalizer": 0.3,
            "telegram_formatter": 0.5,
            "seo_optimizer": 0.4,
        },
        "pipeline": {
            "limit": 10,
            "min_relevance": 5,
            "min_publish_score": 7,
            "max_retries": 3,
            "publish_telegraph": True,
            "publish_telegram": False,
        },
        "telegraph": {
            "paragraphs_per_image": 3,
            "max_content_length": 64000,
        },
    }


# =============================================================================
# Чтение / запись конфигурации
# =============================================================================

def load_config() -> Dict[str, Any]:
    """
    Загрузить конфигурацию.

    Приоритет:
    1. config/settings.json (основной)
    2. config/models.yaml (если json не найден и pyyaml доступен)
    3. Значения по умолчанию
    """
    merged = _defaults()

    # Пробуем JSON (основной)
    if SETTINGS_JSON.exists():
        try:
            with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            _deep_update(merged, data)
            return merged
        except Exception as e:
            st.warning(f"⚠️ Ошибка чтения {SETTINGS_JSON}: {e}")

    # Пробуем YAML (fallback)
    if HAS_YAML and MODELS_YAML.exists():
        try:
            with open(MODELS_YAML, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            _deep_update(merged, data)
            return merged
        except Exception:
            pass

    return merged


def save_config(data: Dict[str, Any]) -> bool:
    """
    Сохранить конфигурацию в JSON и (опционально) в YAML.

    JSON сохраняется всегда — он не требует pyyaml.
    YAML обновляется если pyyaml доступен (для совместимости с pipeline).
    """
    try:
        SETTINGS_JSON.parent.mkdir(parents=True, exist_ok=True)

        # 1. Сохраняем JSON (основной)
        with open(SETTINGS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 2. Обновляем YAML если доступен (для models_config.py)
        if HAS_YAML:
            _sync_to_yaml(data)

        return True
    except Exception as e:
        st.error(f"❌ Ошибка сохранения: {e}")
        return False


def _sync_to_yaml(data: Dict[str, Any]):
    """
    Синхронизировать настройки в models.yaml.

    Обновляет только секции из UI, не трогает остальные.
    """
    try:
        if MODELS_YAML.exists():
            with open(MODELS_YAML, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
        else:
            yaml_data = {}

        for key in ("defaults", "ollama", "temperatures", "pipeline", "telegraph"):
            if key in data:
                yaml_data[key] = data[key]

        MODELS_YAML.parent.mkdir(parents=True, exist_ok=True)
        with open(MODELS_YAML, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)
    except Exception as e:
        st.warning(f"⚠️ YAML синхронизация: {e}")


def _deep_update(base: dict, override: dict):
    """Рекурсивное обновление словаря."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# =============================================================================
# Работа с Ollama
# =============================================================================

@st.cache_data(ttl=30)
def get_ollama_models() -> List[str]:
    """Получить список установленных моделей из Ollama API."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            return sorted([m["name"] for m in resp.json().get("models", [])])
    except Exception:
        pass
    return []


def ollama_is_up() -> bool:
    """Проверить доступность Ollama."""
    try:
        return requests.head(f"{OLLAMA_URL}/", timeout=3).status_code == 200
    except Exception:
        return False


def pull_ollama_model(model_name: str):
    """
    Скачать модель через Ollama API.

    Returns:
        (success: bool, message: str)
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": model_name, "stream": False},
            timeout=600,
        )
        if resp.status_code == 200:
            return True, "OK"
        else:
            return False, resp.text
    except requests.Timeout:
        return False, (
            "Timeout — модель слишком большая. "
            f"Загрузите вручную: docker exec ollama ollama pull {model_name}"
        )
    except Exception as e:
        return False, str(e)


# =============================================================================
# Главная страница настроек
# =============================================================================

def render_settings_page():
    """Отрисовка страницы ⚙️ Настройки."""

    st.title("⚙️ Настройки")

    cfg = load_config()
    changed = False

    tab_main, tab_temps, tab_pipeline, tab_raw = st.tabs([
        "🤖 Провайдер",
        "🌡️ Температуры",
        "🔧 Pipeline",
        "📄 Конфиг",
    ])

    # =================================================================
    # Вкладка 1: Провайдер и модель
    # =================================================================

    with tab_main:
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Провайдер")

            new_provider = st.selectbox(
                "LLM провайдер",
                PROVIDERS,
                index=(
                    PROVIDERS.index(cfg["defaults"]["provider"])
                    if cfg["defaults"]["provider"] in PROVIDERS else 0
                ),
                help="Ollama — локальный, Groq — бесплатный облачный, OpenRouter — платный",
            )
            if new_provider != cfg["defaults"]["provider"]:
                cfg["defaults"]["provider"] = new_provider
                changed = True

            new_fallback = st.toggle(
                "Fallback при ошибке",
                value=cfg["defaults"]["enable_fallback"],
                help="При ошибке переключиться на следующий провайдер",
            )
            if new_fallback != cfg["defaults"]["enable_fallback"]:
                cfg["defaults"]["enable_fallback"] = new_fallback
                changed = True

        with col_right:
            st.subheader("Модель Ollama")

            is_up = ollama_is_up()
            st.caption("🟢 Ollama доступен" if is_up else "🔴 Ollama недоступен")

            models = get_ollama_models()
            current = cfg["ollama"]["model"]

            if models:
                if current not in models:
                    models.insert(0, current)
                new_model = st.selectbox("Выбрать модель", models, index=models.index(current))
            else:
                new_model = st.text_input("Модель (ручной ввод)", value=current)

            if new_model != current:
                cfg["ollama"]["model"] = new_model
                changed = True

            new_ctx = st.number_input(
                "Context length",
                min_value=2048, max_value=131072, step=1024,
                value=cfg["ollama"].get("context_length", 32768),
            )
            if new_ctx != cfg["ollama"].get("context_length", 32768):
                cfg["ollama"]["context_length"] = new_ctx
                changed = True

        # --- Скачивание модели ---
        st.divider()
        st.subheader("📥 Скачать модель")

        pull_col1, pull_col2 = st.columns([3, 1])
        with pull_col1:
            pull_name = st.text_input(
                "Имя модели для скачивания",
                placeholder="glm4:latest, qwen2.5:7b, llama3.1:8b ...",
                key="pull_model_name",
            )
        with pull_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            pull_clicked = st.button("📥 Pull", type="primary", use_container_width=True)

        if pull_clicked and pull_name:
            with st.spinner(f"Скачиваю {pull_name}... Это может занять несколько минут."):
                ok, msg = pull_ollama_model(pull_name.strip())
            if ok:
                st.success(f"✅ Модель {pull_name} загружена!")
                get_ollama_models.clear()
                st.rerun()
            else:
                st.error(f"❌ Ошибка: {msg}")

        if models:
            with st.expander(f"📦 Установленные модели ({len(models)})"):
                for m in models:
                    icon = "✅" if m == cfg["ollama"]["model"] else "　"
                    st.text(f"{icon} {m}")

    # =================================================================
    # Вкладка 2: Температуры
    # =================================================================

    with tab_temps:
        st.subheader("🌡️ Температуры агентов")
        st.caption("0.0 → точно, предсказуемо　　1.0 → творчески, разнообразно")

        temps = cfg.get("temperatures", {})
        cols = st.columns(2)

        for i, (key, label) in enumerate(AGENT_LABELS.items()):
            with cols[i % 2]:
                cur = float(temps.get(key, 0.5))
                new_val = st.slider(label, 0.0, 1.5, cur, 0.05, key=f"t_{key}")
                if abs(new_val - cur) > 0.01:
                    cfg["temperatures"][key] = round(new_val, 2)
                    changed = True

        st.divider()
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            if st.button("🎯 Точный (0.1)", use_container_width=True):
                for k in cfg["temperatures"]:
                    cfg["temperatures"][k] = 0.1
                changed = True
        with pc2:
            if st.button("⚖️ Стандарт", use_container_width=True):
                cfg["temperatures"] = copy.deepcopy(_defaults()["temperatures"])
                changed = True
        with pc3:
            if st.button("🎨 Творческий (0.8)", use_container_width=True):
                for k in cfg["temperatures"]:
                    cfg["temperatures"][k] = 0.8
                changed = True

    # =================================================================
    # Вкладка 3: Pipeline
    # =================================================================

    with tab_pipeline:
        st.subheader("🔧 Параметры Pipeline")
        p = cfg["pipeline"]

        col1, col2 = st.columns(2)

        with col1:
            new_limit = st.number_input("Лимит статей", 1, 100, p["limit"])
            if new_limit != p["limit"]:
                p["limit"] = new_limit
                changed = True

            new_rel = st.slider("Мин. релевантность", 1, 10, p["min_relevance"])
            if new_rel != p["min_relevance"]:
                p["min_relevance"] = new_rel
                changed = True

            new_retries = st.number_input("Макс. повторов", 0, 10, p["max_retries"])
            if new_retries != p["max_retries"]:
                p["max_retries"] = new_retries
                changed = True

        with col2:
            new_score = st.slider("Мин. score для публикации", 1, 10, p["min_publish_score"])
            if new_score != p["min_publish_score"]:
                p["min_publish_score"] = new_score
                changed = True

            new_tg_ph = st.toggle("📝 Telegraph", value=p["publish_telegraph"])
            if new_tg_ph != p["publish_telegraph"]:
                p["publish_telegraph"] = new_tg_ph
                changed = True

            new_tg = st.toggle("📱 Telegram", value=p["publish_telegram"])
            if new_tg != p["publish_telegram"]:
                p["publish_telegram"] = new_tg
                changed = True

        st.divider()
        tph = cfg.get("telegraph", {})
        new_ppi = st.number_input("Абзацев между изображениями (Telegraph)", 1, 10, tph.get("paragraphs_per_image", 3))
        if new_ppi != tph.get("paragraphs_per_image", 3):
            cfg.setdefault("telegraph", {})["paragraphs_per_image"] = new_ppi
            changed = True

        st.divider()
        run1, run2 = st.columns(2)
        with run1:
            if st.button("▶️ Запустить Pipeline", type="primary", use_container_width=True):
                _run_pipeline(cfg)
        with run2:
            if st.button("📊 Статус", use_container_width=True):
                _show_status()

    # =================================================================
    # Вкладка 4: Сырой конфиг
    # =================================================================

    with tab_raw:
        st.subheader("📄 Конфигурация (JSON)")
        st.caption("Прямое редактирование. Сохранение перезаписывает файл.")

        raw = json.dumps(cfg, ensure_ascii=False, indent=2)
        edited = st.text_area("JSON", raw, height=500, key="raw_editor")

        if st.button("💾 Сохранить конфиг"):
            try:
                parsed = json.loads(edited)
                if isinstance(parsed, dict) and save_config(parsed):
                    st.success("✅ Сохранено")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("❌ Невалидный формат")
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON ошибка: {e}")

        st.divider()
        st.caption(f"📁 JSON: `{SETTINGS_JSON}`")
        if HAS_YAML:
            st.caption(f"📁 YAML: `{MODELS_YAML}` (синхронизируется)")
        else:
            st.caption("⚠️ pyyaml не установлен — YAML синхронизация отключена")

    # =================================================================
    # Сохранение (сайдбар)
    # =================================================================

    st.sidebar.divider()

    if changed:
        st.sidebar.warning("⚠️ Несохранённые изменения")

    if st.sidebar.button("💾 Сохранить настройки", type="primary", use_container_width=True):
        if save_config(cfg):
            os.environ["LLM_PROVIDER"] = cfg["defaults"]["provider"]
            os.environ["OLLAMA_MODEL"] = cfg["ollama"]["model"]
            if not cfg["defaults"]["enable_fallback"]:
                os.environ["ENABLE_FALLBACK"] = "false"
            else:
                os.environ.pop("ENABLE_FALLBACK", None)

            st.sidebar.success("✅ Сохранено")
            st.cache_data.clear()
            st.rerun()

    with st.sidebar.expander("🔍 Текущий env"):
        st.code(
            f"LLM_PROVIDER={os.getenv('LLM_PROVIDER', '-')}\n"
            f"OLLAMA_MODEL={os.getenv('OLLAMA_MODEL', '-')}\n"
            f"ENABLE_FALLBACK={os.getenv('ENABLE_FALLBACK', '-')}\n"
            f"OLLAMA_URL={OLLAMA_URL}",
            language="bash",
        )


# =============================================================================
# Pipeline API
# =============================================================================

def _run_pipeline(cfg: Dict[str, Any]):
    """Запустить pipeline через REST API."""
    p = cfg["pipeline"]
    payload = {
        "limit": p["limit"],
        "provider": cfg["defaults"]["provider"],
        "no_fallback": not cfg["defaults"]["enable_fallback"],
        "min_relevance": p["min_relevance"],
        "min_publish_score": p["min_publish_score"],
        "max_retries": p["max_retries"],
        "publish_telegraph": p["publish_telegraph"],
        "publish_telegram": p["publish_telegram"],
    }
    try:
        resp = requests.post(f"{API_URL}/api/v1/pipeline/run", json=payload, timeout=10)
        if resp.status_code == 200:
            st.success(f"✅ {resp.json().get('message', 'Запущен')}")
        elif resp.status_code == 409:
            st.warning("⚠️ Pipeline уже запущен")
        else:
            st.error(f"❌ {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"❌ API: {e}")


def _show_status():
    """Показать статус pipeline."""
    try:
        resp = requests.get(f"{API_URL}/api/v1/pipeline/status", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("is_running"):
                st.info("🔄 Pipeline работает...")
            elif data.get("last_run"):
                last = data["last_run"]
                s = last.get("status", "?")
                st.info(f"{'✅' if s == 'completed' else '❌'} {s} — {last.get('duration_seconds', '?')}с")
            else:
                st.info("💤 Ещё не запускался")
    except Exception as e:
        st.error(f"❌ {e}")