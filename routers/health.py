# ==============================================================================
# 1. Health Router Tests (GET /api/v1/health)
# ==============================================================================

"""
Health Router

Provides endpoints for health checks and system status
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import warnings

warnings.filterwarnings("ignore")

# Global router
health_router = APIRouter()


# Endpoint to check health of the service
@health_router.get("/health")
async def get_health_check():
    try:
        return JSONResponse(
            content={"message": "Service Health is Good."},
            status_code=200,
        )
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)
