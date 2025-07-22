"""
Main API router for v1 endpoints
"""
from fastapi import APIRouter

from app.api.v1.endpoints import auth, usage

api_router = APIRouter()

# Include authentication endpoints
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])

# Include usage and subscription endpoints
api_router.include_router(usage.router, prefix="/usage", tags=["usage"])

# Placeholder for future endpoint routers
# from app.api.v1.endpoints import locations, notes, photos, voice, query, media

# api_router.include_router(locations.router, prefix="/locations", tags=["locations"])
# api_router.include_router(notes.router, prefix="/notes", tags=["notes"])
# api_router.include_router(photos.router, prefix="/photos", tags=["photos"])
# api_router.include_router(voice.router, prefix="/voice", tags=["voice"])
# api_router.include_router(query.router, prefix="/query", tags=["query"])
# api_router.include_router(media.router, prefix="/media", tags=["media"])

@api_router.get("/")
async def api_info():
    return {"message": "ChronoTrail API v1", "status": "active"}