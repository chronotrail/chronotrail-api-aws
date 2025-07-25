"""
Main API router for v1 endpoints
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    locations,
    media,
    notes,
    photos,
    query,
    usage,
    voice,
)

api_router = APIRouter()

# Include authentication endpoints
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])

# Include usage and subscription endpoints
api_router.include_router(usage.router, prefix="/usage", tags=["usage"])

# Include location visits endpoints
api_router.include_router(locations.router, prefix="/locations", tags=["locations"])

# Include text notes endpoints
api_router.include_router(notes.router, prefix="/notes", tags=["notes"])

# Include photo upload endpoints
api_router.include_router(photos.router, prefix="/photos", tags=["photos"])

# Include voice upload endpoints
api_router.include_router(voice.router, prefix="/voice", tags=["voice"])

# Include query endpoint
api_router.include_router(query.router, prefix="/query", tags=["query"])

# Include media retrieval endpoints
api_router.include_router(media.router, prefix="/media", tags=["media"])


@api_router.get("/")
async def api_info():
    return {"message": "ChronoTrail API v1", "status": "active"}
