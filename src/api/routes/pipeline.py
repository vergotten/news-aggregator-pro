# -*- coding: utf-8 -*-
"""
API Route: Pipeline - запуск полного конвейера через HTTP.

Путь: src/api/routes/pipeline.py

Позволяет n8n (или любому HTTP-клиенту) запускать pipeline по расписанию:

    POST /api/v1/pipeline/run
    {
        "limit": 10,
        "hubs": "",
        "provider": "ollama",
        "publish_telegraph": true,
        "publish_telegram": true,
        "min_publish_score": 7,
        "min_relevance": 5
    }

    GET /api/v1/pipeline/status
    → текущий статус pipeline (idle / running / last result)
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


# =============================================================================
# Schemas
# =============================================================================

class PipelineRunRequest(BaseModel):
    """Параметры запуска pipeline."""
    limit: int = Field(default=10, ge=1, le=100, description="Количество статей")
    hubs: str = Field(default="", description="Хабы через запятую")
    provider: Optional[str] = Field(default=None, description="LLM провайдер (ollama/openrouter/groq/google)")
    no_fallback: bool = Field(default=False, description="Отключить fallback")
    strategy: Optional[str] = Field(default=None, description="Стратегия моделей")
    min_relevance: int = Field(default=5, ge=1, le=10, description="Мин. релевантность для Qdrant")
    publish_telegraph: bool = Field(default=False, description="Публиковать на Telegraph")
    publish_telegram: bool = Field(default=False, description="Отправлять в Telegram")
    min_publish_score: int = Field(default=7, ge=1, le=10, description="Мин. score для публикации")
    max_retries: int = Field(default=3, ge=0, le=10, description="Макс. повторов")
    verbose: bool = Field(default=False)
    debug: bool = Field(default=False)


class PipelineRunResponse(BaseModel):
    """Ответ на запуск pipeline."""
    status: str
    message: str
    correlation_id: Optional[str] = None
    started_at: Optional[str] = None


class PipelineStatusResponse(BaseModel):
    """Текущий статус pipeline."""
    is_running: bool
    last_run: Optional[dict] = None


# =============================================================================
# State (in-memory, singleton per worker)
# =============================================================================

class _PipelineState:
    """Состояние pipeline (один запуск одновременно)."""

    def __init__(self):
        self.is_running: bool = False
        self.last_run: Optional[dict] = None
        self._lock = asyncio.Lock()

    async def try_start(self) -> bool:
        """Попробовать начать. False если уже запущен."""
        async with self._lock:
            if self.is_running:
                return False
            self.is_running = True
            return True

    async def finish(self, result: dict):
        """Завершить запуск."""
        async with self._lock:
            self.is_running = False
            self.last_run = result


_state = _PipelineState()


# =============================================================================
# Background task
# =============================================================================

async def _run_pipeline_task(params: PipelineRunRequest):
    """
    Фоновая задача запуска pipeline.

    Запускается через asyncio.create_task() — работает в event loop FastAPI,
    но не блокирует другие запросы благодаря await.
    """
    started_at = datetime.utcnow()

    try:
        from run_full_pipeline import full_pipeline

        logger.info(f"[Pipeline API] Starting: limit={params.limit}, provider={params.provider}")

        await full_pipeline(
            limit=params.limit,
            hubs=params.hubs,
            verbose=params.verbose,
            min_relevance=params.min_relevance,
            debug=params.debug,
            provider=params.provider,
            strategy=params.strategy,
            no_fallback=params.no_fallback,
            max_retries=params.max_retries,
            publish_telegraph=params.publish_telegraph,
            publish_telegram=params.publish_telegram,
            min_publish_score=params.min_publish_score,
        )

        result = {
            "status": "completed",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
            "duration_seconds": round(time.time() - started_at.timestamp(), 1),
            "params": params.model_dump(),
        }

        logger.info(f"[Pipeline API] Completed in {result['duration_seconds']}s")

    except Exception as e:
        logger.error(f"[Pipeline API] Failed: {e}", exc_info=True)
        result = {
            "status": "failed",
            "error": str(e),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
            "duration_seconds": round(time.time() - started_at.timestamp(), 1),
            "params": params.model_dump(),
        }

    await _state.finish(result)


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(
    params: PipelineRunRequest,
):
    """
    Запустить полный pipeline обработки статей.

    - Парсинг Habr
    - AI обработка (классификация, релевантность, тизер, заголовок, нормализация)
    - Сохранение в БД + Qdrant
    - (опционально) Публикация на Telegraph + Telegram

    Запускается в фоне. Только один запуск одновременно.

    Для n8n:
        HTTP Request Node → POST http://api:8000/api/v1/pipeline/run
        Body: {"limit": 10, "publish_telegraph": true, "publish_telegram": true}
    """
    if not await _state.try_start():
        raise HTTPException(
            status_code=409,
            detail="Pipeline уже запущен. Дождитесь завершения или проверьте /api/v1/pipeline/status"
        )

    # create_task — не блокирует event loop, /status будет отвечать
    asyncio.create_task(_run_pipeline_task(params))

    return PipelineRunResponse(
        status="started",
        message=f"Pipeline запущен: {params.limit} статей, "
                f"telegraph={'ON' if params.publish_telegraph else 'OFF'}, "
                f"telegram={'ON' if params.publish_telegram else 'OFF'}",
        started_at=datetime.utcnow().isoformat(),
    )


@router.get("/status", response_model=PipelineStatusResponse)
async def pipeline_status():
    """
    Проверить статус pipeline.

    Для n8n:
        HTTP Request Node → GET http://api:8000/api/v1/pipeline/status
    """
    return PipelineStatusResponse(
        is_running=_state.is_running,
        last_run=_state.last_run,
    )


@router.post("/stop")
async def stop_pipeline():
    """
    Остановить pipeline (пометить как не запущенный).

    Не прерывает текущую обработку, но сбрасывает блокировку
    если pipeline завис.
    """
    if not _state.is_running:
        return {"status": "not_running", "message": "Pipeline не запущен"}

    await _state.finish({
        "status": "force_stopped",
        "finished_at": datetime.utcnow().isoformat(),
    })

    return {"status": "stopped", "message": "Блокировка сброшена"}