from fastapi import APIRouter, Response
from typing import List
from app.models.access_log import AccessLog, AccessLogResponse
from app.services.access_service import access_service

router = APIRouter(prefix="/access", tags=["access"])

@router.post("/log")
async def log_access(access_log: AccessLog):
    """Registra un evento de entrada o salida"""
    await access_service.log_access(access_log)
    return {"message": "Registro de acceso guardado exitosamente"}

@router.get("/person/{person_id}", response_model=List[AccessLogResponse])
async def get_person_logs(person_id: str):
    """Obtiene el historial de entradas y salidas de una persona"""
    logs = await access_service.get_person_logs(person_id)
    if not logs:
        return Response(content="Persona no encontrada", status_code=404)
    return logs

@router.get("/current", response_model=List[AccessLogResponse])
async def get_current_people():
    """Obtiene la lista de personas que están actualmente en el recinto"""
    return await access_service.get_current_people() 