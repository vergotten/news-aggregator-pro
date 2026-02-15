# src/domain/entities/n8n_workflow.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class WorkflowStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class ExecutionStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELED = "canceled"
    WAITING = "waiting"


@dataclass
class N8nWorkflow:
    id: str
    name: str
    status: WorkflowStatus
    description: Optional[str] = None
    tags: List[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version_id: Optional[str] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    connections: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class Schedule:
    id: str
    workflow_id: str
    name: str
    cron_expression: str
    is_active: bool = True
    timezone: str = "UTC"
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class WorkflowExecution:
    id: str
    workflow_id: str
    status: ExecutionStatus
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    mode: str = "manual"
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_of: Optional[str] = None
    retry_success_id: Optional[str] = None