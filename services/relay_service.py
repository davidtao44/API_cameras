import httpx
import asyncio
from typing import Dict, List, Optional, Any
from models.relay import (
    RelayConfig, RelayState, RelayCommand, RelayResponse,
    RelayStatus, RelayType, HTTPRelayConfig
)
from config.relay_config import get_all_relay_configs
import logging

logger = logging.getLogger(__name__)

class ShellyRelayService:
    """Servicio para controlar relés Shelly Plus 1 PM mediante API HTTP y RPC"""
    
    def __init__(self):
        self.relays: Dict[str, RelayConfig] = {}
        self.timeout = 10.0
        # Cargar configuración estática de relés
        self._load_static_relay_config()
    
    def _load_static_relay_config(self):
        """Cargar configuración estática de relés desde el archivo de configuración"""
        try:
            static_configs = get_all_relay_configs()
            for relay_id, config in static_configs.items():
                self.relays[relay_id] = config
                logger.info(f"Relé estático {relay_id} cargado: {config.name}")
            logger.info(f"Se cargaron {len(static_configs)} relés desde la configuración estática")
        except Exception as e:
            logger.error(f"Error cargando configuración estática de relés: {e}")
    
    def add_relay(self, relay_id: str, config: RelayConfig) -> bool:
        """Agregar un relé a la configuración"""
        try:
            self.relays[relay_id] = config
            logger.info(f"Relé {relay_id} agregado exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error agregando relé {relay_id}: {e}")
            return False
    
    def remove_relay(self, relay_id: str) -> bool:
        """Remover un relé de la configuración"""
        if relay_id in self.relays:
            del self.relays[relay_id]
            logger.info(f"Relé {relay_id} removido exitosamente")
            return True
        return False
    
    def get_relay_config(self, relay_id: str) -> Optional[RelayConfig]:
        """Obtener configuración de un relé"""
        return self.relays.get(relay_id)
    
    def list_relays(self) -> Dict[str, RelayConfig]:
        """Listar todos los relés configurados"""
        return self.relays.copy()
    
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
        url = f"http://{ip}/rpc"
        
        # Construir frame RPC
        rpc_frame = {
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            result = await self._make_http_request(url, "POST", rpc_frame)
            
            if "error" in result:
                error = result["error"]
                raise Exception(f"Error RPC {error.get('code', 'unknown')}: {error.get('message', 'Unknown error')}")
            
            return result.get("result", {})
        except Exception as e:
            logger.error(f"Error enviando comando RPC a {ip}: {e}")
            raise
    
    async def set_relay_state(self, relay_id: str, command: RelayCommand) -> RelayResponse:
        """Controlar estado del relé"""
        if relay_id not in self.relays:
            return RelayResponse(
                success=False,
                message=f"Relé {relay_id} no encontrado",
                relay_id=relay_id
            )
        
        config = self.relays[relay_id]
        
        if config.relay_type != RelayType.HTTP:
            return RelayResponse(
                success=False,
                message=f"Tipo de relé no soportado: {config.relay_type}",
                relay_id=relay_id
            )
        
        # Convertir configuración a HTTPRelayConfig si es un diccionario
        http_config = config.config
        if isinstance(http_config, dict):
            try:
                http_config = HTTPRelayConfig(**http_config)
            except Exception as e:
                return RelayResponse(
                    success=False,
                    message=f"Configuración HTTP inválida: {str(e)}",
                    relay_id=relay_id
                )
        elif not isinstance(http_config, HTTPRelayConfig):
            return RelayResponse(
                success=False,
                message="Configuración HTTP inválida",
                relay_id=relay_id
            )
        
        try:
            # Usar RPC para comandos más avanzados
            if command.action in ["on", "off"]:
                params = {
                    "id": http_config.channel,
                    "on": command.action == "on"
                }
                
                if command.timer_seconds:
                    params["toggle_after"] = command.timer_seconds
                
                result = await self._send_rpc_command(
                    http_config.ip_address,
                    "Switch.Set",
                    params,
                    http_config.auth
                )
                
                return RelayResponse(
                    success=True,
                    message=f"Relé {relay_id} {'encendido' if command.action == 'on' else 'apagado'} exitosamente",
                    relay_id=relay_id,
                    previous_state=result.get("was_on", False)
                )
            
            elif command.action == "toggle":
                result = await self._send_rpc_command(
                    http_config.ip_address,
                    "Switch.Toggle",
                    {"id": http_config.channel},
                    http_config.auth
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
        if relay_id not in self.relays:
            return RelayResponse(
                success=False,
                message=f"Relé {relay_id} no encontrado",
                relay_id=relay_id
            )
        
        config = self.relays[relay_id]
        
        if config.relay_type != RelayType.HTTP:
            return RelayResponse(
                success=False,
                message=f"Tipo de relé no soportado: {config.relay_type}",
                relay_id=relay_id
            )
        
        # Convertir configuración a HTTPRelayConfig si es un diccionario
        http_config = config.config
        if isinstance(http_config, dict):
            try:
                http_config = HTTPRelayConfig(**http_config)
            except Exception as e:
                return RelayResponse(
                    success=False,
                    message=f"Configuración HTTP inválida: {str(e)}",
                    relay_id=relay_id
                )
        elif not isinstance(http_config, HTTPRelayConfig):
            return RelayResponse(
                success=False,
                message="Configuración HTTP inválida",
                relay_id=relay_id
            )
        
        try:
            # Obtener estado usando RPC
            result = await self._send_rpc_command(
                http_config.ip_address,
                "Switch.GetStatus",
                {"id": http_config.channel},
                http_config.auth
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
        if relay_id not in self.relays:
            return RelayResponse(
                success=False,
                message=f"Relé {relay_id} no encontrado",
                relay_id=relay_id
            )
        
        config = self.relays[relay_id]
        
        # Convertir configuración a HTTPRelayConfig si es un diccionario
        http_config = config.config
        if isinstance(http_config, dict):
            try:
                http_config = HTTPRelayConfig(**http_config)
            except Exception as e:
                return RelayResponse(
                    success=False,
                    message=f"Configuración HTTP inválida: {str(e)}",
                    relay_id=relay_id
                )
        elif not isinstance(http_config, HTTPRelayConfig):
            return RelayResponse(
                success=False,
                message="Configuración HTTP inválida",
                relay_id=relay_id
            )
        
        try:
            # Obtener información del dispositivo
            result = await self._send_rpc_command(
                http_config.ip_address,
                "Shelly.GetDeviceInfo",
                {},
                http_config.auth
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