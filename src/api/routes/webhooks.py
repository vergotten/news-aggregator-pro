# -*- coding: utf-8 -*-
"""
Webhook Routes для n8n и внешних интеграций.

Путь: src/api/routes/webhooks.py
"""

from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/n8n")
async def n8n_webhook(request: Request):
    """Принять webhook от n8n."""
    data = await request.json()
    print(f"Received from n8n: {data}")

    result = {
        "status": "processed",
        "received": data,
        "timestamp": datetime.utcnow().isoformat()
    }

    return JSONResponse(content=result)