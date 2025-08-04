from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import Response
from typing import List, Dict, Any
from models.alert import AlertLog, AlertResponse
from models.auth import UserResponse
from auth.dependencies import get_current_active_user
from services.alert_service import get_alert_service
import logging

router = APIRouter(
    prefix="/alertas",
    tags=["alertas"]
)

@router.get("/", response_model=List[AlertResponse])
async def obtener_alertas(current_user: UserResponse = Depends(get_current_active_user)):
    """Obtener todas las alertas del sistema"""
    return get_alert_service().get_all_alerts()

@router.get("/{alert_id}", response_model=Dict[str, Any])
async def obtener_alerta(alert_id: str, current_user: UserResponse = Depends(get_current_active_user)):
    """Obtener una alerta específica por ID"""
    return get_alert_service().get_alert_by_id(alert_id)

@router.post("/", response_model=Dict[str, Any])
async def crear_alerta(alert: AlertLog, current_user: UserResponse = Depends(get_current_active_user)):
    """Crear una nueva alerta"""
    return await get_alert_service().create_alert(alert)

@router.get("/{alert_id}/image", response_model=Dict[str, Any])
async def obtener_imagen_alerta(alert_id: str, current_user: UserResponse = Depends(get_current_active_user)):
    """Obtener solo la imagen de una alerta específica"""
    return get_alert_service().get_alert_image(alert_id)