from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from pydantic import BaseModel
from auth.dependencies import get_current_active_user
from models.auth import UserResponse
from surveillance.access_control_service import access_control_service

router = APIRouter(prefix="/access-control", tags=["access-control"])

class AccessControlConfig(BaseModel):
    """Configuración del control de acceso"""
    require_all_identified: Optional[bool] = None
    min_confidence_threshold: Optional[float] = None
    max_unknown_persons: Optional[int] = None
    access_timeout: Optional[float] = None

@router.get("/status/{camera_id}")
async def get_access_status(camera_id: str, current_user: UserResponse = Depends(get_current_active_user)) -> Dict[str, Any]:
    """Obtiene el estado actual del control de acceso para una cámara específica"""
    try:
        result = access_control_service.get_access_status(camera_id)
        
        if result['status'] == 'no_data':
            raise HTTPException(status_code=404, detail=result['message'])
        elif result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado de acceso: {str(e)}")

@router.get("/evaluate/{camera_id}")
async def evaluate_access(camera_id: str, current_user: UserResponse = Depends(get_current_active_user)) -> Dict[str, Any]:
    """Evalúa si se debe conceder acceso basado en las condiciones:
    1. Más de una persona detectada
    2. Número de rostros mayor que número de personas  
    3. Todos los rostros identificados (no desconocidos)
    """
    try:
        access_result = access_control_service.evaluate_access(camera_id)
        
        if not access_result:
            raise HTTPException(status_code=404, detail="No hay datos de procesamiento disponibles para esta cámara")
        
        return {
            "status": "success",
            "camera_id": camera_id,
            "access_granted": access_result.access_granted,
            "evaluation_details": {
                "total_persons": access_result.total_persons,
                "identified_persons": access_result.identified_persons,
                "unknown_persons": access_result.unknown_persons,
                "identified_names": access_result.identified_names,
                "identification_percentage": access_result.identification_percentage,
                "confidence_level": access_result.confidence_level,
                "timestamp": access_result.timestamp.isoformat()
            },
            "message": "Acceso concedido" if access_result.access_granted else "Acceso denegado"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando acceso: {str(e)}")



