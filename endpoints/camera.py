from fastapi import APIRouter, Response, Depends
from fastapi.responses import StreamingResponse
from services.camera_service import generate_stream
from config.camera_config import CAMERAS
from auth.service_dependencies import camera_service_token
from auth.dependencies import get_current_active_user
from models.auth import UserResponse
from typing import Union, Dict, Any

router = APIRouter(prefix="/stream", tags=["camera"])

@router.get("/{camera_id}")
def video_feed(
    camera_id: str,
    # current_user: UserResponse = Depends(get_current_active_user)
):
    """Stream de video - requiere autenticación de usuario"""
    if camera_id not in CAMERAS:
        return Response(content="Cámara no encontrada", status_code=404)
    return StreamingResponse(
        generate_stream(CAMERAS[camera_id], camera_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@router.get("/service/{camera_id}")
def video_feed_service(
    camera_id: str,
    service_token: Dict[str, Any] = Depends(camera_service_token)
):
    """Stream de video para servicios - requiere service token"""
    if camera_id not in CAMERAS:
        return Response(content="Cámara no encontrada", status_code=404)
    return StreamingResponse(
        generate_stream(CAMERAS[camera_id], camera_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@router.get("/info/{camera_id}")
def camera_info(
    camera_id: str,
    service_token: Dict[str, Any] = Depends(camera_service_token)
):
    """Información de cámara - solo service tokens"""
    if camera_id not in CAMERAS:
        return Response(content="Cámara no encontrada", status_code=404)
    
    return {
        "camera_id": camera_id,
        "url": CAMERAS[camera_id],
        "status": "active",
        "accessed_by_service": service_token["service_name"],
        "permissions": service_token["permissions"]
    }