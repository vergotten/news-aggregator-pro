# src/application/services/n8n_service.py
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..domain.entities.n8n_workflow import (
    N8nWorkflow, Schedule, WorkflowExecution,
    WorkflowStatus, ExecutionStatus
)
from ...infrastructure.n8n.n8n_client import N8nClient


class N8nService:
    def __init__(self):
        self.client = N8nClient()

    async def get_dashboard_data(self) -> Dict[str, Any]:
        async with self.client:
            workflows = await self.client.get_workflows()
            schedules = await self.client.get_schedules()
            executions = await self.client.get_executions(limit=100)

            active_workflows = [w for w in workflows if w.status == WorkflowStatus.ACTIVE]
            recent_executions = [e for e in executions if e.started_at and
                                 (datetime.utcnow() - e.started_at).days <= 7]

            return {
                "total_workflows": len(workflows),
                "active_workflows": len(active_workflows),
                "total_schedules": len(schedules),
                "active_schedules": len([s for s in schedules if s.is_active]),
                "recent_executions": len(recent_executions),
                "success_rate": self._calculate_success_rate(recent_executions),
                "workflows": workflows,
                "schedules": schedules,
                "executions": executions[:20]  # Последние 20
            }

    async def toggle_workflow(self, workflow_id: str) -> bool:
        async with self.client:
            workflow = await self.client.get_workflow(workflow_id)
            if not workflow:
                return False

            if workflow.status == WorkflowStatus.ACTIVE:
                return await self.client.deactivate_workflow(workflow_id)
            else:
                return await self.client.activate_workflow(workflow_id)

    async def execute_workflow_manually(self, workflow_id: str, data: Optional[Dict] = None) -> str:
        async with self.client:
            return await self.client.execute_workflow(workflow_id, data)

    async def create_schedule(self, schedule_data: Dict[str, Any]) -> bool:
        schedule = Schedule(
            id="",  # Будет сгенерирован n8n
            workflow_id=schedule_data["workflow_id"],
            name=schedule_data["name"],
            cron_expression=schedule_data["cron_expression"],
            is_active=schedule_data.get("is_active", True),
            timezone=schedule_data.get("timezone", "UTC"),
            parameters=schedule_data.get("parameters", {})
        )
        async with self.client:
            return await self.client.create_schedule(schedule)

    async def update_schedule(self, schedule_id: str, schedule_data: Dict[str, Any]) -> bool:
        async with self.client:
            schedules = await self.client.get_schedules()
            schedule = next((s for s in schedules if s.id == schedule_id), None)
            if not schedule:
                return False

            # Обновляем поля
            schedule.name = schedule_data.get("name", schedule.name)
            schedule.cron_expression = schedule_data.get("cron_expression", schedule.cron_expression)
            schedule.is_active = schedule_data.get("is_active", schedule.is_active)
            schedule.timezone = schedule_data.get("timezone", schedule.timezone)
            schedule.parameters = schedule_data.get("parameters", schedule.parameters)

            return await self.client.update_schedule(schedule)

    async def delete_schedule(self, schedule_id: str) -> bool:
        async with self.client:
            return await self.client.delete_schedule(schedule_id)

    def _calculate_success_rate(self, executions: List[WorkflowExecution]) -> float:
        if not executions:
            return 0.0
        successful = len([e for e in executions if e.status == ExecutionStatus.SUCCESS])
        return round((successful / len(executions)) * 100, 2)