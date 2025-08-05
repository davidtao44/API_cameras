from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from pydantic import BaseModel
from auth.dependencies import get_current_active_user
from models.auth import UserResponse

# Importar el servicio de tracking mejorado
from surveillance.enhanced_frame_processing_service import enhanced_frame_processing_service
from surveillance.person_tracking_service import person_tracking_service

router = APIRouter(prefix="/tracking", tags=["tracking"])

class TrackingConfig(BaseModel):
    """Configuración del sistema de tracking"""
    tracking_mode: str  # "basic" or "enhanced"
    enable_tracking: bool = True
    face_persistence_time: int = 10  # segundos
    track_max_age: int = 15  # segundos
    max_distance_threshold: int = 100  # píxeles
    iou_threshold: float = 0.3

@router.get("/status")
async def get_tracking_status(
    current_user: UserResponse = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Obtiene el estado actual del sistema de tracking"""
    try:
        # Obtener estadísticas del servicio mejorado
        processing_stats = enhanced_frame_processing_service.get_processing_stats()
        
        # Obtener configuración actual del tracking
        tracking_config = {
            "face_persistence_time": person_tracking_service.face_persistence_time,
            "track_max_age": person_tracking_service.track_max_age,
            "max_distance_threshold": person_tracking_service.max_distance_threshold,
            "iou_threshold": person_tracking_service.iou_threshold
        }
        
        return {
            "status": "success",
            "tracking_mode": processing_stats.get("tracking_mode", "enhanced"),
            "tracking_enabled": processing_stats.get("tracking_enabled", True),
            "active_cameras": processing_stats.get("active_cameras", []),
            "tracking_config": tracking_config,
            "processing_stats": processing_stats,
            "message": "Sistema de tracking funcionando correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado del tracking: {str(e)}")

@router.post("/configure")
async def configure_tracking(
    config: TrackingConfig,
    current_user: UserResponse = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Configura el sistema de tracking"""
    try:
        # Validar modo de tracking
        if config.tracking_mode not in ["basic", "enhanced"]:
            raise HTTPException(status_code=400, detail="tracking_mode debe ser 'basic' o 'enhanced'")
        
        # Configurar el servicio de procesamiento mejorado
        enhanced_frame_processing_service.set_tracking_mode(config.tracking_mode)
        enhanced_frame_processing_service.enable_tracking = config.enable_tracking
        
        # Configurar parámetros del tracking
        person_tracking_service.face_persistence_time = config.face_persistence_time
        person_tracking_service.track_max_age = config.track_max_age
        person_tracking_service.max_distance_threshold = config.max_distance_threshold
        person_tracking_service.iou_threshold = config.iou_threshold
        
        return {
            "status": "success",
            "message": f"Tracking configurado en modo {config.tracking_mode}",
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error configurando tracking: {str(e)}")

@router.post("/mode/{mode}")
async def set_tracking_mode(
    mode: str,
    current_user: UserResponse = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Cambia el modo de tracking (básico o mejorado)"""
    try:
        if mode not in ["basic", "enhanced"]:
            raise HTTPException(status_code=400, detail="Modo debe ser 'basic' o 'enhanced'")
        
        enhanced_frame_processing_service.set_tracking_mode(mode)
        
        return {
            "status": "success",
            "message": f"Modo de tracking cambiado a: {mode}",
            "tracking_mode": mode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cambiando modo de tracking: {str(e)}")

@router.post("/enable")
async def enable_tracking(
    current_user: UserResponse = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Activa el sistema de tracking"""
    try:
        enhanced_frame_processing_service.enable_tracking = True
        return {
            "status": "success",
            "message": "Sistema de tracking activado",
            "tracking_enabled": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activando tracking: {str(e)}")

@router.post("/disable")
async def disable_tracking(
    current_user: UserResponse = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Desactiva el sistema de tracking"""
    try:
        enhanced_frame_processing_service.enable_tracking = False
        return {
            "status": "success",
            "message": "Sistema de tracking desactivado",
            "tracking_enabled": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error desactivando tracking: {str(e)}")

@router.get("/stats/{camera_id}")
async def get_camera_tracking_stats(
    camera_id: str,
    current_user: UserResponse = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Obtiene estadísticas de tracking para una cámara específica"""
    try:
        stats = person_tracking_service.get_tracking_stats(camera_id)
        return {
            "status": "success",
            "camera_id": camera_id,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")

@router.post("/cleanup/{camera_id}")
async def cleanup_camera_tracking(
    camera_id: str,
    current_user: UserResponse = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Limpia todos los tracks de una cámara específica"""
    try:
        person_tracking_service.cleanup_camera(camera_id)
        return {
            "status": "success",
            "message": f"Tracks limpiados para la cámara {camera_id}",
            "camera_id": camera_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error limpiando tracks: {str(e)}")