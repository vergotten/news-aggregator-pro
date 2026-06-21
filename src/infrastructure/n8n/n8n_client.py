# src/infrastructure/n8n/n8n_client.py
import aiohttp
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from ..config.settings import get_settings
from ...domain.entities.n8n_workflow import (
    N8nWorkflow, Schedule, WorkflowExecution,
    WorkflowStatus, ExecutionStatus
)


class N8nClient:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.N8N_API_URL.rstrip('/')
        self.api_key = self.settings.N8N_API_KEY
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with.")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        async with self.session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def get_workflows(self) -> List[N8nWorkflow]:
        data = await self._request("GET", "api/v1/workflows")
        return [
            N8nWorkflow(
                id=wf["id"],
                name=wf["name"],
                status=WorkflowStatus(wf["active"]),
                description=wf.get("description"),
                tags=wf.get("tags", []),
                created_at=datetime.fromisoformat(wf["createdAt"]) if wf.get("createdAt") else None,
                updated_at=datetime.fromisoformat(wf["updatedAt"]) if wf.get("updatedAt") else None,
                version_id=wf.get("versionId"),
                nodes=wf.get("nodes", []),
                connections=wf.get("connections", {})
            )
            for wf in data.get("data", [])
        ]

    async def get_workflow(self, workflow_id: str) -> Optional[N8nWorkflow]:
        try:
            data = await self._request("GET", f"api/v1/workflows/{workflow_id}")
            return N8nWorkflow(
                id=data["id"],
                name=data["name"],
                status=WorkflowStatus(data["active"]),
                description=data.get("description"),
                tags=data.get("tags", []),
                created_at=datetime.fromisoformat(data["createdAt"]) if data.get("createdAt") else None,
                updated_at=datetime.fromisoformat(data["updatedAt"]) if data.get("updatedAt") else None,
                version_id=data.get("versionId"),
                nodes=data.get("nodes", []),
                connections=data.get("connections", {})
            )
        except Exception:
            return None

    async def activate_workflow(self, workflow_id: str) -> bool:
        try:
            await self._request("POST", f"api/v1/workflows/{workflow_id}/activate")
            return True
        except Exception:
            return False

    async def deactivate_workflow(self, workflow_id: str) -> bool:
        try:
            await self._request("POST", f"api/v1/workflows/{workflow_id}/deactivate")
            return True
        except Exception:
            return False

    async def execute_workflow(self, workflow_id: str, data: Optional[Dict] = None) -> str:
        payload = {"workflowData": {}} if data is None else {"data": data}
        result = await self._request("POST", f"api/v1/workflows/{workflow_id}/execute", json=payload)
        return result.get("executionId")

    async def get_executions(self, workflow_id: Optional[str] = None, limit: int = 50) -> List[WorkflowExecution]:
        params = {"limit": limit}
        if workflow_id:
            params["workflowId"] = workflow_id

        data = await self._request("GET", "api/v1/executions", params=params)
        return [
            WorkflowExecution(
                id=ex["id"],
                workflow_id=ex["workflowId"],
                status=ExecutionStatus(ex["finished"] if ex.get("finished") else "running"),
                started_at=datetime.fromisoformat(ex["startedAt"]) if ex.get("startedAt") else None,
                stopped_at=datetime.fromisoformat(ex["stoppedAt"]) if ex.get("stoppedAt") else None,
                mode=ex.get("mode", "manual"),
                data=ex.get("data"),
                error=ex.get("error"),
                retry_of=ex.get("retryOf"),
                retry_success_id=ex.get("retrySuccessId")
            )
            for ex in data.get("data", [])
        ]

    async def get_schedules(self) -> List[Schedule]:
        data = await self._request("GET", "rest/schedules")
        return [
            Schedule(
                id=sched["id"],
                workflow_id=sched["workflowId"],
                name=sched["name"],
                cron_expression=sched["cronExpression"],
                is_active=sched["active"],
                timezone=sched.get("timezone", "UTC"),
                next_run=datetime.fromisoformat(sched["nextRun"]) if sched.get("nextRun") else None,
                last_run=datetime.fromisoformat(sched["lastRun"]) if sched.get("lastRun") else None,
                created_at=datetime.fromisoformat(sched["createdAt"]) if sched.get("createdAt") else None,
                updated_at=datetime.fromisoformat(sched["updatedAt"]) if sched.get("updatedAt") else None,
                parameters=sched.get("data", {})
            )
            for sched in data.get("data", [])
        ]

    async def create_schedule(self, schedule: Schedule) -> bool:
        payload = {
            "workflowId": schedule.workflow_id,
            "name": schedule.name,
            "cronExpression": schedule.cron_expression,
            "active": schedule.is_active,
            "timezone": schedule.timezone,
            "data": schedule.parameters or {}
        }
        try:
            await self._request("POST", "rest/schedules", json=payload)
            return True
        except Exception:
            return False

    async def update_schedule(self, schedule: Schedule) -> bool:
        payload = {
            "name": schedule.name,
            "cronExpression": schedule.cron_expression,
            "active": schedule.is_active,
            "timezone": schedule.timezone,
            "data": schedule.parameters or {}
        }
        try:
            await self._request("PATCH", f"rest/schedules/{schedule.id}", json=payload)
            return True
        except Exception:
            return False

    async def delete_schedule(self, schedule_id: str) -> bool:
        try:
            await self._request("DELETE", f"rest/schedules/{schedule_id}")
            return True
        except Exception:
            return False