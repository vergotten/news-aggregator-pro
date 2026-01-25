# -*- coding: utf-8 -*-
"""
Configuration - конфигурация моделей и системы.

Компоненты:
- ModelsConfig: Конфигурация AI моделей с профилями
- AgentType: Перечисление типов агентов
- get_models_config: Получение глобального инстанса конфигурации
"""

from src.config.models_config import (
    ModelsConfig,
    AgentType,
    AgentConfig,
    ProfileConfig,
    get_models_config,
)

__all__ = [
    'ModelsConfig',
    'AgentType',
    'AgentConfig',
    'ProfileConfig',
    'get_models_config',
]
