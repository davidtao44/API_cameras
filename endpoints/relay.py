from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import aiohttp
import asyncio
from typing import Optional

router = APIRouter(prefix="/relay", tags=["relay"])

class RelayRequest(BaseModel):
    ip: str
    relay_id: int
    state: bool  # True para encender, False para apagar
    timeout: Optional[int] = 5  # Timeout en segundos

class RelayResponse(BaseModel):
    success: bool
    message: str
    ip: str
    relay_id: int
    state: bool

@router.post("/switch")
async def switch_relay(request: RelayRequest):
    """
    Controla el estado de un relé mediante petición HTTP
    
    Args:
        request: Datos del relé (IP, ID, estado)
    
    Returns:
        Respuesta con el resultado de la operación
    """
    try:
        # Construir la URL para el relé
        state_str = "true" if request.state else "false"
        url = f"http://{request.ip}/rpc/Switch.Set?id={request.relay_id}&on={state_str}"
        
        # Realizar la petición HTTP
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=request.timeout)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    response_data = await response.text()
                    return RelayResponse(
                        success=True,
                        message=f"Relé {request.relay_id} {'encendido' if request.state else 'apagado'} correctamente",
                        ip=request.ip,
                        relay_id=request.relay_id,
                        state=request.state
                    )
                else:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Error en la respuesta del relé: {response.status}"
                    )
                    
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Timeout al conectar con el relé en {request.ip}"
        )
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error de conexión: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}"
        )

@router.get("/status/{ip}/{relay_id}")
async def get_relay_status(ip: str, relay_id: int, timeout: int = 5):
    """
    Obtiene el estado actual de un relé
    
    Args:
        ip: Dirección IP del relé
        relay_id: ID del relé
        timeout: Timeout en segundos
    
    Returns:
        Estado actual del relé
    """
    try:
        # Construir la URL para obtener el estado
        url = f"http://{ip}/rpc/Switch.GetStatus?id={relay_id}"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    response_data = await response.json()
                    return {
                        "success": True,
                        "ip": ip,
                        "relay_id": relay_id,
                        "state": response_data.get("output", False),
                        "data": response_data
                    }
                else:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Error en la respuesta del relé: {response.status}"
                    )
                    
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Timeout al conectar con el relé en {ip}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )

@router.post("/toggle")
async def toggle_relay(ip: str, relay_id: int, timeout: int = 5):
    """
    Alterna el estado de un relé (encendido/apagado)
    
    Args:
        ip: Dirección IP del relé
        relay_id: ID del relé
        timeout: Timeout en segundos
    
    Returns:
        Nuevo estado del relé
    """
    try:
        # Primero obtener el estado actual
        current_status = await get_relay_status(ip, relay_id, timeout)
        current_state = current_status["state"]
        
        # Alternar el estado
        new_state = not current_state
        
        # Crear la petición para cambiar el estado
        request = RelayRequest(
            ip=ip,
            relay_id=relay_id,
            state=new_state,
            timeout=timeout
        )
        
        return await switch_relay(request)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al alternar relé: {str(e)}"
        )