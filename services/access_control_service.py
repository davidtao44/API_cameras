import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import threading
import time
from concurrent.futures import ThreadPoolExecutor

class AccessControlService:
    def __init__(self):
        self.config = {
            "relay_ip": "172.16.2.47",
            "relay_id": 0,
            "access_duration": 5,
            "cooldown_period": 20,
            "require_all_faces_known": True
        }
        self.last_activation = None
        self.relay_active = False
        self.relay_timer = None
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="access_control")
        
    async def evaluate_access(self, camera_id: str, detected_names: List[str], timestamp: datetime) -> Dict:
        """Evalúa si se debe conceder acceso"""
        # Verificaciones de seguridad
        if not detected_names:
            return self._create_response(False, False, "No se detectaron rostros", detected_names)
            
        if self.config["require_all_faces_known"] and "Desconocido" in detected_names:
            return self._create_response(False, False, "Rostros no reconocidos detectados", detected_names)
            
        if self._is_in_cooldown():
            remaining = self._get_cooldown_remaining()
            return self._create_response(False, False, f"En período de enfriamiento ({remaining}s restantes)", detected_names)
            
        if self.relay_active:
            return self._create_response(True, False, "Acceso ya activo", detected_names)
        
        # Conceder acceso
        success = await self._activate_relay()
        if success:
            self.last_activation = timestamp
            return self._create_response(True, True, f"Acceso concedido por {self.config['access_duration']}s", detected_names)
        else:
            return self._create_response(False, False, "Error al activar relé", detected_names)
    
    def _create_response(self, access_granted: bool, relay_activated: bool, message: str, detected_names: List[str]) -> Dict:
        return {
            "access_granted": access_granted,
            "relay_activated": relay_activated,
            "message": message,
            "detected_names": detected_names
        }
    
    def _is_in_cooldown(self) -> bool:
        if not self.last_activation:
            return False
        elapsed = (datetime.now() - self.last_activation).total_seconds()
        return elapsed < self.config["cooldown_period"]
    
    def _get_cooldown_remaining(self) -> int:
        if not self.last_activation:
            return 0
        elapsed = (datetime.now() - self.last_activation).total_seconds()
        remaining = self.config["cooldown_period"] - elapsed
        return max(0, int(remaining))
    
    async def _activate_relay(self) -> bool:
        """Activa el relé de forma asíncrona"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                url = "http://localhost:8000/relay/switch"
                data = {
                    "ip": self.config["relay_ip"],
                    "relay_id": self.config["relay_id"],
                    "state": True,
                    "timeout": 3
                }
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.relay_active = True
                        self._schedule_deactivation()
                        return True
                    return False
        except Exception as e:
            print(f"Error activando relé: {e}")
            return False
    
    def _schedule_deactivation(self):
        """Programa la desactivación del relé"""
        if self.relay_timer:
            self.relay_timer.cancel()
        
        self.relay_timer = threading.Timer(
            self.config["access_duration"],
            self._deactivate_relay_sync
        )
        self.relay_timer.start()
    
    def _deactivate_relay_sync(self):
        """Desactiva el relé de forma síncrona (para threading.Timer)"""
        asyncio.run(self._deactivate_relay())
    
    async def _deactivate_relay(self) -> bool:
        """Desactiva el relé"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                url = "http://localhost:8000/relay/switch"
                data = {
                    "ip": self.config["relay_ip"],
                    "relay_id": self.config["relay_id"],
                    "state": False,
                    "timeout": 3
                }
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.relay_active = False
                        return True
                    return False
        except Exception as e:
            print(f"Error desactivando relé: {e}")
            return False
    
    async def get_config(self) -> Dict:
        return self.config.copy()
    
    async def update_config(self, new_config: Dict):
        self.config.update(new_config)
    
    async def get_status(self) -> Dict:
        return {
            "relay_active": self.relay_active,
            "last_activation": self.last_activation.isoformat() if self.last_activation else None,
            "in_cooldown": self._is_in_cooldown(),
            "cooldown_remaining": self._get_cooldown_remaining(),
            "config": self.config
        }
    
    async def manual_override(self, activate: bool, duration: Optional[int] = None) -> Dict:
        """Activación/desactivación manual del relé"""
        if activate:
            if duration:
                old_duration = self.config["access_duration"]
                self.config["access_duration"] = duration
            
            success = await self._activate_relay()
            
            if duration:
                self.config["access_duration"] = old_duration
            
            return {
                "success": success,
                "message": "Relé activado manualmente" if success else "Error al activar relé",
                "duration": duration or self.config["access_duration"]
            }
        else:
            success = await self._deactivate_relay()
            if self.relay_timer:
                self.relay_timer.cancel()
            return {
                "success": success,
                "message": "Relé desactivado manualmente" if success else "Error al desactivar relé"
            }

# Instancia global del servicio
access_control_service = AccessControlService()