import httpx
import asyncio
from typing import Dict, List, Optional, Any
from models.relay import (
    RelayConfig, RelayState, RelayCommand, RelayResponse,
    RelayStatus, RelayType, HTTPRelayConfig
)
from config.relay_config import get_relay_config, get_all_relay_configs, relay_exists
import logging

logger = logging.getLogger(__name__)

class ShellyRelayService:
    """Servicio para controlar relés Shelly Plus 1 PM mediante API HTTP y RPC"""
    
    def __init__(self):
        self.timeout = 10.0
    
    def get_relay_config(self, relay_id: str) -> Optional[RelayConfig]:
        """Obtener configuración de un relé"""
        try:
            return get_relay_config(relay_id)
        except ValueError:
            return None
    
    def list_relays(self) -> Dict[str, RelayConfig]:
        """Listar todos los relés configurados"""
        return get_all_relay_configs()
    
    async def _make_http_request(self, url: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
        """Realizar petición HTTP a dispositivo Shelly"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(url)
                elif method.upper() == "POST":
                    response = await client.post(url, json=data)
                else:
                    raise ValueError(f"Método HTTP no soportado: {method}")
                
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException:
            raise Exception(f"Timeout al conectar con el dispositivo: {url}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"Error HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise Exception(f"Error de conexión: {str(e)}")
    
    async def _send_rpc_command(self, ip: str, method: str, params: Dict[str, Any], auth: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Enviar comando RPC a dispositivo Shelly Gen2+"""
        url = f"http://{ip}/rpc/{method}"
        
        try:
            result = await self._make_http_request(url, "POST", params)
            return result
        except Exception as e:
            logger.error(f"Error enviando comando RPC a {ip}: {e}")
            raise
    
    async def _send_http_relay_command(self, ip: str, channel: int, command: str, timer: Optional[int] = None, auth: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Enviar comando HTTP directo al endpoint /relay/id"""
        url = f"http://{ip}/relay/{channel}?turn={command}"
        
        if timer:
            url += f"&timer={timer}"
        
        try:
            return await self._make_http_request(url, "GET")
        except Exception as e:
            logger.error(f"Error enviando comando HTTP a {ip}: {e}")
            raise
    
    async def set_relay_state(self, relay_id: str, command: RelayCommand) -> RelayResponse:
        """Controlar estado del relé"""
        if not relay_exists(relay_id):
            return RelayResponse(
                success=False,
                message=f"Relé {relay_id} no encontrado",
                relay_id=relay_id
            )
        
        config = self.get_relay_config(relay_id)
        
        if config.relay_type != RelayType.HTTP:
            return RelayResponse(
                success=False,
                message=f"Tipo de relé no soportado: {config.relay_type}",
                relay_id=relay_id
            )
        
        http_config = config.config
        
        # Extraer configuración del diccionario
        ip_address = http_config.get("ip_address")
        channel = http_config.get("channel", 0)
        auth = http_config.get("auth")
        use_rpc = http_config.get("use_rpc", True)
        
        if not ip_address:
            return RelayResponse(
                success=False,
                message="Dirección IP no configurada",
                relay_id=relay_id
            )
        
        try:
            # Usar RPC para comandos más avanzados
            if command.action in ["on", "off"]:
                params = {
                    "id": channel,
                    "on": command.action == "on"
                }
                
                if command.timer_seconds:
                    params["toggle_after"] = command.timer_seconds
                
                result = await self._send_rpc_command(
                    ip_address,
                    "Switch.Set",
                    params,
                    auth
                )
                
                return RelayResponse(
                    success=True,
                    message=f"Relé {relay_id} {'encendido' if command.action == 'on' else 'apagado'} exitosamente",
                    relay_id=relay_id,
                    previous_state=result.get("was_on", False)
                )
            
            elif command.action == "toggle":
                result = await self._send_rpc_command(
                    ip_address,
                    "Switch.Toggle",
                    {"id": channel},
                    auth
                )
                
                return RelayResponse(
                    success=True,
                    message=f"Relé {relay_id} alternado exitosamente",
                    relay_id=relay_id,
                    previous_state=result.get("was_on", False)
                )
            
            else:
                return RelayResponse(
                    success=False,
                    message=f"Acción no soportada: {command.action}",
                    relay_id=relay_id
                )
        
        except Exception as e:
            logger.error(f"Error controlando relé {relay_id}: {e}")
            return RelayResponse(
                success=False,
                message=f"Error: {str(e)}",
                relay_id=relay_id
            )
    
    async def get_relay_status(self, relay_id: str) -> RelayResponse:
        """Obtener estado actual del relé"""
        if not relay_exists(relay_id):
            return RelayResponse(
                success=False,
                message=f"Relé {relay_id} no encontrado",
                relay_id=relay_id
            )
        
        config = self.get_relay_config(relay_id)
        
        if config.relay_type != RelayType.HTTP:
            return RelayResponse(
                success=False,
                message=f"Tipo de relé no soportado: {config.relay_type}",
                relay_id=relay_id
            )
        
        http_config = config.config
        
        # Extraer configuración del diccionario
        ip_address = http_config.get("ip_address")
        channel = http_config.get("channel", 0)
        auth = http_config.get("auth")
        
        if not ip_address:
            return RelayResponse(
                success=False,
                message="Dirección IP no configurada",
                relay_id=relay_id
            )
        
        try:
            # Obtener estado usando RPC
            result = await self._send_rpc_command(
                ip_address,
                "Switch.GetStatus",
                {"id": channel},
                auth
            )
            
            # Crear objeto RelayState con la información obtenida
            relay_state = RelayState(
                relay_id=relay_id,
                is_on=result.get("output", False),
                power_consumption=result.get("apower", 0.0),
                voltage=result.get("voltage", 0.0),
                current=result.get("current", 0.0),
                energy_total=result.get("aenergy", {}).get("total", 0.0),
                temperature=result.get("temperature", {}).get("tC", 0.0),
                has_timer=result.get("timer_started_at") is not None,
                timer_remaining=result.get("timer_remaining", 0),
                source=result.get("source", "unknown")
            )
            
            return RelayResponse(
                success=True,
                message="Estado obtenido exitosamente",
                relay_id=relay_id,
                state=relay_state
            )
        
        except Exception as e:
            logger.error(f"Error obteniendo estado del relé {relay_id}: {e}")
            return RelayResponse(
                success=False,
                message=f"Error: {str(e)}",
                relay_id=relay_id
            )
    
    async def get_device_info(self, relay_id: str) -> RelayResponse:
        """Obtener información del dispositivo Shelly"""
        if not relay_exists(relay_id):
            return RelayResponse(
                success=False,
                message=f"Relé {relay_id} no encontrado",
                relay_id=relay_id
            )
        
        config = self.get_relay_config(relay_id)
        http_config = config.config
        
        # Extraer configuración del diccionario
        ip_address = http_config.get("ip_address")
        auth = http_config.get("auth")
        
        if not ip_address:
            return RelayResponse(
                success=False,
                message="Dirección IP no configurada",
                relay_id=relay_id
            )
        
        try:
            # Obtener información del dispositivo
            result = await self._send_rpc_command(
                ip_address,
                "Shelly.GetDeviceInfo",
                {},
                auth
            )
            
            return RelayResponse(
                success=True,
                message="Información del dispositivo obtenida exitosamente",
                relay_id=relay_id,
                device_info=result
            )
        
        except Exception as e:
            logger.error(f"Error obteniendo información del dispositivo {relay_id}: {e}")
            return RelayResponse(
                success=False,
                message=f"Error: {str(e)}",
                relay_id=relay_id
            )

# Instancia global del servicio
relay_service = ShellyRelayService()