# -*- coding: utf-8 -*-
"""
AI Services - сервисы для обработки контента с использованием LLM.

Компоненты:
- agents: Специализированные AI агенты
- orchestrator: Координатор для запуска pipeline обработки
"""

from src.application.ai_services.orchestrator import (
    AIOrchestrator,
    create_orchestrator,
    create_local_orchestrator,
    create_cloud_orchestrator,
)

# Алиас для обратной совместимости
ContentOrchestrator = AIOrchestrator

__all__ = [
    'AIOrchestrator',
    'create_orchestrator',
    'create_local_orchestrator',
    'create_cloud_orchestrator',
]