from fastapi import APIRouter

from api.app.api.routes.jobs import router as jobs_router
from api.app.api.routes.chat import router as chat_router
from api.app.api.routes.assets import router as assets_router

api_router = APIRouter()
api_router.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
api_router.include_router(chat_router, prefix="/chat/sessions", tags=["chat"])
api_router.include_router(assets_router, prefix="/assets", tags=["assets"])
