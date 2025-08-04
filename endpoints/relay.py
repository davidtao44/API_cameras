from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Optional
from models.relay import (
    RelayConfig, RelayCommand, RelayResponse, RelayListResponse,
    RelayConfigResponse, RelayState, RelayType, HTTPRelayConfig
)
from models.auth import UserResponse
from auth.dependencies import get_current_active_user
from services.relay_service import relay_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/relay", tags=["relay"])



@router.get("/list", response_model=RelayListResponse)
async def list_relays(
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Listar todos los relés configurados en el sistema.
    Muestra información básica de cada relé desde la configuración estática.
    """
    try:
        from config.relay_config import get_relay_list
        
        relay_list = get_relay_list()
        
        return RelayListResponse(
            success=True,
            message=f"Se encontraron {len(relay_list)} relés configurados",
            relays=relay_list,
            total_count=len(relay_list)
        )
    
    except Exception as e:
        logger.error(f"Error en list_relays: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )

@router.get("/config/{relay_id}", response_model=RelayConfigResponse)
async def get_relay_config(
    relay_id: str,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Obtener la configuración de un relé específico.
    """
    try:
        config = relay_service.get_relay_config(relay_id)
        
        if config:
            return RelayConfigResponse(
                success=True,
                message=f"Configuración del relé {relay_id} obtenida exitosamente",
                relay_id=relay_id,
                config=config
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Relé {relay_id} no encontrado"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en get_relay_config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )

@router.post("/control/{relay_id}", response_model=RelayResponse)
async def control_relay(
    relay_id: str,
    command: RelayCommand,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Controlar el estado de un relé (encender, apagar, alternar).
    """
    try:
        response = await relay_service.set_relay_state(relay_id, command)
        
        if not response.success:
            if "no encontrado" in response.message:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=response.message
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=response.message
                )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en control_relay: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )

@router.get("/status/{relay_id}", response_model=RelayResponse)
async def get_relay_status(
    relay_id: str,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Obtener el estado actual de un relé (encendido/apagado, consumo, etc.).
    """
    try:
        response = await relay_service.get_relay_status(relay_id)
        
        if not response.success:
            if "no encontrado" in response.message:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=response.message
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=response.message
                )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en get_relay_status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )

@router.get("/device-info/{relay_id}", response_model=RelayResponse)
async def get_device_info(
    relay_id: str,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Obtener información del dispositivo Shelly (modelo, firmware, etc.).
    """
    try:
        response = await relay_service.get_device_info(relay_id)
        
        if not response.success:
            if "no encontrado" in response.message:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=response.message
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=response.message
                )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en get_device_info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )



# Endpoints de conveniencia para acciones rápidas
@router.get("/on/{relay_id}", response_model=RelayResponse)
async def turn_relay_on(
    relay_id: str,
    timer_seconds: Optional[int] = None,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Encender un relé. Endpoint de conveniencia.
    
    timer_seconds: Tiempo en segundos para apagar automáticamente (opcional)
    """
    command = RelayCommand(action="on", timer_seconds=timer_seconds)
    return await control_relay(relay_id, command, current_user)

@router.get("/off/{relay_id}", response_model=RelayResponse)
async def turn_relay_off(
    relay_id: str,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Apagar un relé. Endpoint de conveniencia.
    """
    command = RelayCommand(action="off")
    return await control_relay(relay_id, command, current_user)

@router.get("/toggle/{relay_id}", response_model=RelayResponse)
async def toggle_relay(
    relay_id: str,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Alternar estado de un relé (encendido <-> apagado). Endpoint de conveniencia.
    """
    command = RelayCommand(action="toggle")
    return await control_relay(relay_id, command, current_user)