from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import Response
from models.auth import UserResponse
from auth.dependencies import get_current_active_user
from services.image_service import get_image_service
import logging

router = APIRouter(
    prefix="/images",
    tags=["imágenes"]
)

@router.get("/{visitor_id}")
async def get_visitor_image(visitor_id: str, current_user: UserResponse = Depends(get_current_active_user)) -> Response:
    """Obtener imagen de visitante como archivo de imagen"""
    return get_image_service().get_visitor_image(visitor_id)