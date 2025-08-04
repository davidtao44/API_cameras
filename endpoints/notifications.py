from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse
from models.auth import UserResponse
from auth.dependencies import get_current_active_user
from services.notification_service import get_notification_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/notifications",
    tags=["notificaciones"]
)

@router.get("/events")
async def stream_notifications(request: Request, current_user: UserResponse = Depends(get_current_active_user)):
    """Endpoint para Server-Sent Events - permite al frontend recibir notificaciones en tiempo real"""
    logger.info("Cliente conectándose al stream de notificaciones")
    
    notification_service = get_notification_service()
    
    return StreamingResponse(
        notification_service.create_event_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@router.get("/status")
async def get_notification_status(current_user: UserResponse = Depends(get_current_active_user)):
    """Obtener el estado del servicio de notificaciones"""
    notification_service = get_notification_service()
    return {
        "status": "active",
        "connected_clients": len(notification_service.clients),
        "message": "Servicio de notificaciones funcionando correctamente"
    }