from fastapi import APIRouter, Response
from models.schemas import RecognitionStats
from services.stats_service import get_camera_stats

router = APIRouter(prefix="/stats", tags=["statistics"])

@router.get("/{camera_id}", response_model=RecognitionStats)
def get_stats(camera_id: str):
    """Endpoint para obtener estadísticas de reconocimiento"""
    stats = get_camera_stats(camera_id)
    if not stats:
        return Response(content="Cámara no encontrada", status_code=404)
    return stats

@router.get("/people-count/{camera_id}")
def get_people_count(camera_id: str):
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
def get_recognized_people(camera_id: str):
    """Endpoint para obtener la lista de personas reconocidas"""
    stats = get_camera_stats(camera_id)
    if not stats:
        return Response(content="Cámara no encontrada", status_code=404)
    
    return {
        "recognized_people": list(stats["recognized_names"]),
        "count": len(stats["recognized_names"])
    }