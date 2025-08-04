from fastapi import APIRouter, Response, Depends
from models.schemas import RecognitionStats
from services.stats_service import get_camera_stats
from surveillance.enhanced_frame_processing_service import enhanced_frame_processing_service
from auth.dependencies import get_current_active_user
from models.auth import UserResponse

router = APIRouter(prefix="/stats", tags=["statistics"])

@router.get("/{camera_id}", response_model=RecognitionStats)
def get_stats(camera_id: str, current_user: UserResponse = Depends(get_current_active_user)):
    """Endpoint para obtener estadísticas de reconocimiento"""
    stats = get_camera_stats(camera_id)
    if not stats:
        return Response(content="Cámara no encontrada", status_code=404)
    return stats

@router.get("/people-count/{camera_id}")
def get_people_count(camera_id: str, current_user: UserResponse = Depends(get_current_active_user)):
    """Endpoint para obtener el conteo de personas actual"""
    stats = get_camera_stats(camera_id)
    if not stats:
        return Response(content="Cámara no encontrada", status_code=404)
    
    return {
        "current_people": stats["recognized"] + stats["unrecognized"],
        "recognized": stats["recognized"],
        "unrecognized": stats["unrecognized"]
    }

@router.get("/recognized-people/{camera_id}")
def get_recognized_people(camera_id: str, current_user: UserResponse = Depends(get_current_active_user)):
    """Endpoint para obtener la lista de personas reconocidas"""
    stats = get_camera_stats(camera_id)
    if not stats:
        return Response(content="Cámara no encontrada", status_code=404)
    
    return {
        "recognized_people": list(stats["recognized_names"]),
        "count": len(stats["recognized_names"])
    }

@router.get("/frame-processing")
def get_frame_processing_stats(current_user: UserResponse = Depends(get_current_active_user)):
    """Endpoint para obtener estadísticas del procesamiento centralizado de frames"""
    return enhanced_frame_processing_service.get_processing_stats()

@router.get("/frame-processing/{camera_id}")
def get_camera_processing_status(camera_id: str, current_user: UserResponse = Depends(get_current_active_user)):
    """Endpoint para obtener el estado del procesamiento de una cámara específica"""
    result = enhanced_frame_processing_service.get_latest_processing_result(camera_id)
    
    if not result:
        return {
            "camera_id": camera_id,
            "status": "no_recent_processing",
            "message": "No hay resultados recientes de procesamiento"
        }
    
    return {
        "camera_id": camera_id,
        "status": "active",
        "last_processing": result.timestamp.isoformat(),
        "person_count": result.person_count,
        "face_count": result.face_count,
        "detected_names": result.detected_names
    }