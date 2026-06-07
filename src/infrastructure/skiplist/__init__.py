# -*- coding: utf-8 -*-
"""
Skiplist module — управление проблемными URL.
"""

from src.infrastructure.skiplist.skiplist_service import (
    SkiplistService,
    SkipEntry,
    SkipReason,
    get_skiplist_service,
    reset_skiplist_service,
)

__all__ = [
    'SkiplistService',
    'SkipEntry',
    'SkipReason',
    'get_skiplist_service',
    'reset_skiplist_service',
]
