"""
FastAPI Application Entry Point.

Путь: src/main.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import articles
from src.api.routes import webhooks
from src.api.routes import pipeline
from src.infrastructure.config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title="News Aggregator API",
    description="Система агрегации новостей с AI-обработкой",
    version="2.1.0",
    debug=settings.debug
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(articles.router, prefix="/api/v1")
app.include_router(pipeline.router, prefix="/api/v1")
app.include_router(webhooks.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.1.0"
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "News Aggregator API",
        "version": "2.1.0",
        "docs": "/docs",
        "health": "/health",
        "pipeline": "/api/v1/pipeline/run",
    }