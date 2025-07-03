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
            "require_all_faces_known": True,
            "detection_wait_time": 3,  # Nuevo: tiempo de espera para detectar múltiples personas
            "min_detections_for_activation": 1  # Nuevo: mínimo de detecciones antes de evaluar
        }
        self.last_activation = None
        self.relay_active = False
        self.relay_timer = None
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="access_control")
        
        # Nuevos atributos para manejo de detecciones múltiples
        self.pending_detections = {}  # {camera_id: {"names": [], "timestamp": datetime, "timer": Timer}}
        self.detection_lock = threading.Lock()
        
    async def evaluate_access(self, camera_id: str, detected_names: List[str], timestamp: datetime) -> Dict:
        """Evalúa si se debe conceder acceso con tiempo de espera para múltiples personas"""
        # Verificaciones de seguridad básicas
        if not detected_names:
            return self._create_response(False, False, "No se detectaron rostros", detected_names)
            
        if self.config["require_all_faces_known"] and "Desconocido" in detected_names:
            return self._create_response(False, False, "Rostros no reconocidos detectados", detected_names)
            
        if self._is_in_cooldown():
            remaining = self._get_cooldown_remaining()
            return self._create_response(False, False, f"En período de enfriamiento ({remaining}s restantes)", detected_names)
            
        if self.relay_active:
            return self._create_response(True, False, "Acceso ya activo", detected_names)
        
        # Manejar detecciones con tiempo de espera
        return await self._handle_detection_with_wait(camera_id, detected_names, timestamp)
    
    async def _handle_detection_with_wait(self, camera_id: str, detected_names: List[str], timestamp: datetime) -> Dict:
        """Maneja las detecciones con tiempo de espera para múltiples personas"""
        with self.detection_lock:
            # Si ya hay una detección pendiente para esta cámara
            if camera_id in self.pending_detections:
                # Actualizar la lista de nombres detectados (sin duplicados)
                current_names = self.pending_detections[camera_id]["names"]
                for name in detected_names:
                    if name not in current_names:
                        current_names.append(name)
                
                self.pending_detections[camera_id]["names"] = current_names
                self.pending_detections[camera_id]["timestamp"] = timestamp
                
                return self._create_response(
                    False, False, 
                    f"Esperando más detecciones... ({len(current_names)} personas detectadas)", 
                    current_names
                )
            else:
                # Primera detección para esta cámara
                self.pending_detections[camera_id] = {
                    "names": detected_names.copy(),
                    "timestamp": timestamp,
                    "timer": None
                }
                
                # Programar evaluación después del tiempo de espera
                timer = threading.Timer(
                    self.config["detection_wait_time"],
                    self._evaluate_pending_detection,
                    args=[camera_id]
                )
                timer.start()
                self.pending_detections[camera_id]["timer"] = timer
                
                return self._create_response(
                    False, False, 
                    f"Primera detección registrada. Esperando {self.config['detection_wait_time']}s por más personas...", 
                    detected_names
                )
    
    def _evaluate_pending_detection(self, camera_id: str):
        """Evalúa las detecciones pendientes después del tiempo de espera"""
        with self.detection_lock:
            if camera_id not in self.pending_detections:
                return
            
            detection_data = self.pending_detections[camera_id]
            detected_names = detection_data["names"]
            timestamp = detection_data["timestamp"]
            
            # Limpiar la detección pendiente
            del self.pending_detections[camera_id]
        
        # Evaluar si se debe activar el relé
        num_people = len(detected_names)
        print(f"🔍 Evaluando acceso final: {num_people} personas detectadas: {detected_names}")
        
        # Verificar si hay suficientes personas conocidas
        known_people = [name for name in detected_names if name != "Desconocido"]
        
        if len(known_people) >= self.config["min_detections_for_activation"]:
            # Activar relé de forma asíncrona
            asyncio.run(self._activate_relay_final(detected_names, timestamp))
        else:
            print(f"⚠️ No se activó el relé: solo {len(known_people)} personas conocidas (mínimo: {self.config['min_detections_for_activation']})")
    
    async def _activate_relay_final(self, detected_names: List[str], timestamp: datetime):
        """Activa el relé después de la evaluación final"""
        success = await self._activate_relay()
        if success:
            self.last_activation = timestamp
            print(f"✅ Acceso concedido para {len(detected_names)} personas: {detected_names}")
        else:
            print(f"❌ Error al activar relé para: {detected_names}")
    
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
        with self.detection_lock:
            pending_info = {
                camera_id: {
                    "names": data["names"],
                    "timestamp": data["timestamp"].isoformat(),
                    "waiting": True
                }
                for camera_id, data in self.pending_detections.items()
            }
        
        return {
            "relay_active": self.relay_active,
            "last_activation": self.last_activation.isoformat() if self.last_activation else None,
            "in_cooldown": self._is_in_cooldown(),
            "cooldown_remaining": self._get_cooldown_remaining(),
            "pending_detections": pending_info,
            "config": self.config
        }
    
    async def manual_override(self, activate: bool, duration: Optional[int] = None) -> Dict:
        """Activación/desactivación manual del relé"""
        # Cancelar detecciones pendientes en caso de override manual
        if activate:
            with self.detection_lock:
                for camera_id, data in self.pending_detections.items():
                    if data["timer"]:
                        data["timer"].cancel()
                self.pending_detections.clear()
        
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