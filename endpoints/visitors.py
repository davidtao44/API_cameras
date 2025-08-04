from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Dict, Any
from models.visitor import VisitorModel, VisitorResponse, StatusUpdateModel
from models.auth import UserResponse
from auth.dependencies import get_current_active_user
from services.visitor_service import get_visitor_service
import logging

router = APIRouter(
    prefix="/visitantes",
    tags=["visitantes"]
)

@router.get("/registros", response_model=List[VisitorResponse])
async def obtener_registros(current_user: UserResponse = Depends(get_current_active_user)):
    """Obtener todos los registros de visitantes"""
    return get_visitor_service().get_all_visitors()

@router.post("/registros", response_model=Dict[str, Any])
async def crear_visitante(visitor: VisitorModel, current_user: UserResponse = Depends(get_current_active_user)):
    """Crear un nuevo registro de visitante"""
    return get_visitor_service().create_visitor(visitor)

@router.put("/registros/{visitor_id}/estado", response_model=Dict[str, Any])
async def actualizar_estado_visitante(visitor_id: str, status_update: StatusUpdateModel, current_user: UserResponse = Depends(get_current_active_user)):
    """Actualizar el estado de un visitante"""
    return get_visitor_service().update_visitor_status(visitor_id, status_update.estado)

@router.get("/registros/{visitor_id}", response_model=Dict[str, Any])
async def obtener_visitante(visitor_id: str, current_user: UserResponse = Depends(get_current_active_user)):
    """Obtener un visitante específico por ID"""
    return get_visitor_service().get_visitor_by_id(visitor_id)