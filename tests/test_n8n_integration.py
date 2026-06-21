# tests/test_n8n_integration.py
import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime
from src.infrastructure.n8n.n8n_client import N8nClient
from src.application.services.n8n_service import N8nService
from src.domain.entities.n8n_workflow import WorkflowStatus, ExecutionStatus


@pytest.fixture
def mock_n8n_client():
    with patch('src.infrastructure.n8n.n8n_client.aiohttp.ClientSession') as mock_session:
        client = N8nClient()
        client.session = AsyncMock()
        yield client


@pytest.mark.asyncio
async def test_get_workflows(mock_n8n_client):
    mock_response = {
        "data": [
            {
                "id": "123",
                "name": "Test Workflow",
                "active": True,
                "description": "Test",
                "tags": ["test"],
                "createdAt": "2023-01-01T00:00:00Z",
                "updatedAt": "2023-01-01T00:00:00Z",
                "versionId": "v1",
                "nodes": [],
                "connections": {}
            }
        ]
    }
    mock_n8n_client.session.request.return_value.json.return_value = mock_response

    workflows = await mock_n8n_client.get_workflows()
    assert len(workflows) == 1
    assert workflows[0].name == "Test Workflow"
    assert workflows[0].status == WorkflowStatus.ACTIVE


@pytest.mark.asyncio
async def test_execute_workflow(mock_n8n_client):
    mock_response = {"executionId": "exec-123"}
    mock_n8n_client.session.request.return_value.json.return_value = mock_response

    execution_id = await mock_n8n_client.execute_workflow("workflow-123")
    assert execution_id == "exec-123"


@pytest.mark.asyncio
async def test_get_dashboard_data():
    service = N8nService()

    with patch.object(service, 'client') as mock_client:
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get_workflows.return_value = []
        mock_client.get_schedules.return_value = []
        mock_client.get_executions.return_value = []

        data = await service.get_dashboard_data()
        assert "total_workflows" in data
        assert "active_workflows" in data
        assert data["total_workflows"] == 0


def test_service_calculate_success_rate():
    service = N8nService()
    from src.domain.entities.n8n_workflow import WorkflowExecution

    executions = [
        WorkflowExecution(id="1", workflow_id="w1", status=ExecutionStatus.SUCCESS),
        WorkflowExecution(id="2", workflow_id="w1", status=ExecutionStatus.SUCCESS),
        WorkflowExecution(id="3", workflow_id="w1", status=ExecutionStatus.ERROR),
    ]

    rate = service._calculate_success_rate(executions)
    assert rate == 66.67


if __name__ == "__main__":
    pytest.main([__file__])