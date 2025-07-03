from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import asyncio
from services.access_control_service import access_control_service

router = APIRouter(prefix="/access-control", tags=["access-control"])

class AccessControlRequest(BaseModel):
    camera_id: str
    detected_names: List[str]
    timestamp: Optional[datetime] = None

class AccessControlResponse(BaseModel):
    access_granted: bool
    relay_activated: bool
    message: str
    detected_names: List[str]

class AccessControlConfig(BaseModel):
    relay_ip: str
    relay_id: int
    access_duration: int  # segundos
    cooldown_period: int  # segundos
    require_all_faces_known: bool = True

@router.post("/evaluate", response_model=AccessControlResponse)
async def evaluate_access(request: AccessControlRequest):
    """Evalúa si se debe conceder acceso basado en los rostros detectados"""
    try:
        result = await access_control_service.evaluate_access(
            camera_id=request.camera_id,
            detected_names=request.detected_names,
            timestamp=request.timestamp or datetime.now()
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando acceso: {str(e)}")

@router.get("/config")
async def get_access_config():
    """Obtiene la configuración actual del control de acceso"""
    return await access_control_service.get_config()

@router.put("/config")
async def update_access_config(config: AccessControlConfig):
    """Actualiza la configuración del control de acceso"""
    await access_control_service.update_config(config)
    return {"message": "Configuración actualizada exitosamente"}

@router.get("/status")
async def get_access_status():
    """Obtiene el estado actual del sistema de control de acceso"""
    return await access_control_service.get_status()

@router.post("/manual-override")
async def manual_override(activate: bool, duration: Optional[int] = None):
    """Activación manual del relé (override del sistema automático)"""
    result = await access_control_service.manual_override(activate, duration)
    return result