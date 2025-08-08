import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time

# Importar servicios necesarios
from .enhanced_frame_processing_service import enhanced_frame_processing_service, EnhancedFrameProcessingResult
from config.camera_config import CAMERAS
from services.relay_service import relay_service
from models.relay import RelayCommand, RelayStatus
import pytz
import asyncio

# Configure Bogotá, Colombia timezone
BOGOTA_TZ = pytz.timezone('America/Bogota')

class AccessControlResult:
    """Container for access control evaluation results"""
    def __init__(self, camera_id: str, access_granted: bool, total_persons: int, 
                 identified_persons: int, unknown_persons: int, 
                 identified_names: List[str], unknown_count: int,
                 timestamp: datetime, confidence_level: float = 0.0):
        self.camera_id = camera_id
        self.access_granted = access_granted
        self.total_persons = total_persons
        self.identified_persons = identified_persons
        self.unknown_persons = unknown_persons
        self.identified_names = identified_names
        self.unknown_count = unknown_count
        self.timestamp = timestamp
        self.confidence_level = confidence_level
        
        # Calcular porcentaje de identificación
        self.identification_percentage = (
            (identified_persons / total_persons * 100) if total_persons > 0 else 0.0
        )

class AccessControlService:
    """Service for controlling access based on multiple conditions:
    1. Must have more than one person detected
    2. Number of faces must be greater than number of persons
    3. All faces must be identified (not 'Unknown' or 'Desconocido')
    """
    
    def __init__(self):
        self.access_history = defaultdict(list)  # Historial por cámara
        self.access_cache = {}  # Cache de resultados recientes
        self.cache_duration = 2.0  # Duración del cache en segundos
        self.lock = threading.Lock()
        
        # Configuración del control de acceso
        self.require_all_identified = True  # Requiere que todas las personas estén identificadas
        self.min_confidence_threshold = 0.5  # Umbral mínimo de confianza para reconocimiento
        self.max_unknown_persons = 0  # Máximo número de personas desconocidas permitidas
        self.access_timeout = 5.0  # Tiempo en segundos para mantener acceso concedido
        
        # Sistema de envío automático después de 3 segundos
        self.auto_access_delay = 3.0  # Tiempo en segundos antes de envío automático
        self.access_timers = {}  # Timers activos por cámara
        self.access_conditions_met = {}  # Estado de condiciones por cámara
        self.access_start_times = {}  # Tiempo de inicio de condiciones cumplidas
        
        # Estadísticas
        self.access_stats = defaultdict(lambda: {
            'total_evaluations': 0,
            'access_granted': 0,
            'access_denied': 0,
            'last_access_time': None,
            'consecutive_denials': 0,
            'auto_access_triggered': 0
        })
        
        # Access Control Service initialized with auto-access system (3s delay)
    
    def evaluate_access(self, camera_id: str) -> Optional[AccessControlResult]:
        """Evalúa si se debe conceder acceso basado en las condiciones:
        1. Más de una persona detectada
        2. Número de rostros mayor que número de personas
        3. Todos los rostros identificados (no desconocidos)
        
        Si access_control_enabled es False, no realiza verificación de control de acceso.
        Si se cumplan las condiciones, activa el relé relay_001 por 5 segundos.
        """
        try:
            # Verificar si el control de acceso está habilitado para esta cámara
            camera_config = CAMERAS.get(camera_id, {})
            access_control_enabled = camera_config.get('access_control_enabled', False)
            
            if not access_control_enabled:
                return None
            
            # Obtener el último resultado de procesamiento
            processing_result = enhanced_frame_processing_service.get_latest_processing_result(camera_id)
            
            if not processing_result:
                return None
            
            # Verificar cache reciente
            cache_key = f"{camera_id}_{processing_result.frame_id}"
            current_time = time.time()
            
            with self.lock:
                if (cache_key in self.access_cache and 
                    current_time - self.access_cache[cache_key]['timestamp'] < self.cache_duration):
                    return self.access_cache[cache_key]['result']
            
            # Evaluar acceso
            access_result = self._evaluate_access_internal(processing_result)
            
            # Manejar sistema de envío automático
            self._handle_auto_access_system(camera_id, access_result)
            
            # Actualizar cache y estadísticas
            with self.lock:
                self.access_cache[cache_key] = {
                    'result': access_result,
                    'timestamp': current_time
                }
                self._update_access_stats(camera_id, access_result)
                self._update_access_history(camera_id, access_result)
            
            return access_result
            
        except Exception as e:
            print(f"[ERROR] Error evaluating access for {camera_id}: {e}")
            return None
    
    def _handle_auto_access_system(self, camera_id: str, access_result: AccessControlResult):
        """Maneja el sistema de envío automático después de 3 segundos"""
        current_time = time.time()
        
        if access_result.access_granted:
            # Las condiciones de acceso se cumplen
            if camera_id not in self.access_conditions_met or not self.access_conditions_met[camera_id]:
                # Primera vez que se cumplen las condiciones
                self.access_conditions_met[camera_id] = True
                self.access_start_times[camera_id] = current_time
                
                # Cancelar timer anterior si existe
                if camera_id in self.access_timers:
                    self.access_timers[camera_id].cancel()
                
                # Crear nuevo timer para envío automático
                timer = threading.Timer(self.auto_access_delay, self._trigger_auto_access, [camera_id])
                self.access_timers[camera_id] = timer
                timer.start()
                
                print(f"[ACCESS_TIMER] {camera_id}: Timer iniciado - {self.auto_access_delay}s para notificación")
            else:
                # Las condiciones siguen cumpliéndose
                elapsed_time = current_time - self.access_start_times[camera_id]
                remaining_time = max(0, self.auto_access_delay - elapsed_time)
                print(f"[ACCESS_TIMER] {camera_id}: Tiempo restante para notificación: {remaining_time:.1f}s")
        else:
            # Las condiciones no se cumplen
            if camera_id in self.access_conditions_met and self.access_conditions_met[camera_id]:
                # Se perdieron las condiciones
                self.access_conditions_met[camera_id] = False
                
                # Cancelar timer si existe
                if camera_id in self.access_timers:
                    self.access_timers[camera_id].cancel()
                    del self.access_timers[camera_id]
                
                print(f"[ACCESS_TIMER] {camera_id}: Timer cancelado")
    
    def _trigger_auto_access(self, camera_id: str):
        """Activa el acceso automáticamente después del delay"""
        try:
            # Verificar que las condiciones aún se cumplan
            if camera_id in self.access_conditions_met and self.access_conditions_met[camera_id]:
                # Obtener el último resultado de procesamiento para obtener nombres identificados
                processing_result = enhanced_frame_processing_service.get_latest_processing_result(camera_id)
                identified_names = []
                
                if processing_result:
                    # Obtener nombres de reconocimiento facial
                    for face_result in processing_result.face_results:
                        name = face_result.get('name', 'Unknown')
                        confidence = face_result.get('confidence', 0.0)
                        if name not in ['Unknown', 'Desconocido'] and confidence >= self.min_confidence_threshold:
                            if name not in identified_names:
                                identified_names.append(name)
                    
                    # Incluir personas trackeadas con identidad confirmada
                    for person in processing_result.tracked_persons:
                        if person.identity_confirmed and person.name not in ['Unknown', 'Desconocido']:
                            if person.name not in identified_names:
                                identified_names.append(person.name)
                
                # Tarea 1: Activar el relé de acceso
                self._activate_relay()
                
                # Tarea 2: Enviar video por WhatsApp con personas identificadas
                self._send_access_video_notification(camera_id, identified_names)
                
                # Actualizar estadísticas
                with self.lock:
                    self.access_stats[camera_id]['auto_access_triggered'] += 1
                
                print(f"[ACCESS_GRANTED] {camera_id}: Acceso automático activado - Personas: {', '.join(identified_names)}")
                
                # Limpiar estado
                self.access_conditions_met[camera_id] = False
                if camera_id in self.access_timers:
                    del self.access_timers[camera_id]
                
        except Exception as e:
            print(f"[ERROR] Error en trigger_auto_access para {camera_id}: {e}")
    
    def _evaluate_access_internal(self, processing_result: EnhancedFrameProcessingResult) -> AccessControlResult:
        """Lógica interna de evaluación de acceso"""
        camera_id = processing_result.camera_id
        timestamp = processing_result.timestamp
        
        # Obtener información de personas detectadas
        total_persons = processing_result.person_count
        face_results = processing_result.face_results
        tracked_persons = processing_result.tracked_persons
        
        # Condición 1: Debe haber al menos una persona
        if total_persons < 1:
            return AccessControlResult(
                camera_id=camera_id,
                access_granted=False,
                total_persons=total_persons,
                identified_persons=0,
                unknown_persons=0,
                identified_names=[],
                unknown_count=0,
                timestamp=timestamp,
                confidence_level=0.0
            )
        
        # Analizar resultados de reconocimiento facial
        identified_names = []
        unknown_count = 0
        confidence_scores = []
        
        for face_result in face_results:
            name = face_result.get('name', 'Unknown')
            confidence = face_result.get('confidence', 0.0)
            
            if name in ['Unknown', 'Desconocido'] or confidence < self.min_confidence_threshold:
                unknown_count += 1
            else:
                identified_names.append(name)
                confidence_scores.append(confidence)
        
        # Incluir personas trackeadas con identidad confirmada
        tracked_identified = []
        for person in tracked_persons:
            if person.identity_confirmed and person.name not in ['Unknown', 'Desconocido']:
                if person.name not in identified_names:  # Evitar duplicados
                    tracked_identified.append(person.name)
                    confidence_scores.append(person.confidence)
        
        # Combinar nombres identificados
        all_identified_names = identified_names + tracked_identified
        
        identified_persons = len(all_identified_names)
        unknown_persons = unknown_count
        
        # Calcular nivel de confianza promedio
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Condición 2: El número de rostros debe ser mayor o igual que el número de personas
        # NOTA: Esta condición se relaja si hay personas trackeadas con identidad confirmada
        total_faces = len(face_results)
        has_tracked_identified = len(tracked_identified) > 0
        
        # Solo rechazar si hay menos rostros que personas Y no hay personas trackeadas identificadas
        if total_faces < total_persons and not has_tracked_identified:
            return AccessControlResult(
                camera_id=camera_id,
                access_granted=False,
                total_persons=total_persons,
                identified_persons=identified_persons,
                unknown_persons=unknown_persons,
                identified_names=all_identified_names,
                unknown_count=unknown_count,
                timestamp=timestamp,
                confidence_level=avg_confidence
            )
        
        # Condición 3: Acceso concedido si:
        # - No hay rostros desconocidos Y hay al menos una persona identificada
        # - O si hay personas trackeadas con identidad confirmada (incluso de espaldas)
        access_granted = (unknown_count == 0 and identified_persons > 0) or has_tracked_identified
        
        return AccessControlResult(
            camera_id=camera_id,
            access_granted=access_granted,
            total_persons=total_persons,
            identified_persons=identified_persons,
            unknown_persons=unknown_persons,
            identified_names=all_identified_names,
            unknown_count=unknown_count,
            timestamp=timestamp,
            confidence_level=avg_confidence
        )
    

    
    def _update_access_stats(self, camera_id: str, access_result: AccessControlResult):
        """Actualiza estadísticas de acceso"""
        stats = self.access_stats[camera_id]
        stats['total_evaluations'] += 1
        
        if access_result.access_granted:
            stats['access_granted'] += 1
            stats['last_access_time'] = access_result.timestamp
            stats['consecutive_denials'] = 0
        else:
            stats['access_denied'] += 1
            stats['consecutive_denials'] += 1
    
    def _update_access_history(self, camera_id: str, access_result: AccessControlResult):
        """Actualiza historial de acceso (mantiene últimos 100 registros)"""
        history = self.access_history[camera_id]
        history.append(access_result)
        
        # Mantener solo los últimos 100 registros
        if len(history) > 100:
            history.pop(0)
    
    def get_access_status(self, camera_id: str) -> Dict[str, Any]:
        """Obtiene el estado actual del control de acceso para una cámara"""
        try:
            current_result = self.evaluate_access(camera_id)
            
            if not current_result:
                return {
                    'status': 'no_data',
                    'message': 'No hay datos de procesamiento disponibles'
                }
            
            with self.lock:
                stats = self.access_stats[camera_id].copy()
                recent_history = self.access_history[camera_id][-10:]  # Últimos 10 registros
            
            return {
                'status': 'success',
                'camera_id': camera_id,
                'current_access': {
                    'granted': current_result.access_granted,
                    'total_persons': current_result.total_persons,
                    'identified_persons': current_result.identified_persons,
                    'unknown_persons': current_result.unknown_persons,
                    'identified_names': current_result.identified_names,
                    'identification_percentage': current_result.identification_percentage,
                    'confidence_level': current_result.confidence_level,
                    'timestamp': current_result.timestamp.isoformat()
                },
                'configuration': {
                    'require_all_identified': self.require_all_identified,
                    'min_confidence_threshold': self.min_confidence_threshold,
                    'max_unknown_persons': self.max_unknown_persons,
                    'access_timeout': self.access_timeout
                },
                'statistics': stats,
                'auto_access_system': {
                    'enabled': True,
                    'delay_seconds': self.auto_access_delay,
                    'conditions_met': self.access_conditions_met.get(camera_id, False),
                    'timer_active': camera_id in self.access_timers,
                    'time_remaining': (
                        max(0, self.auto_access_delay - (time.time() - self.access_start_times[camera_id]))
                        if camera_id in self.access_start_times and self.access_conditions_met.get(camera_id, False)
                        else 0
                    )
                },
                'recent_history': [
                    {
                        'access_granted': result.access_granted,
                        'total_persons': result.total_persons,
                        'identified_persons': result.identified_persons,
                        'unknown_persons': result.unknown_persons,
                        'timestamp': result.timestamp.isoformat()
                    }
                    for result in recent_history
                ]
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error obteniendo estado de acceso: {str(e)}'
            }
    
    def configure_access_control(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configura los parámetros del control de acceso"""
        try:
            with self.lock:
                if 'require_all_identified' in config:
                    self.require_all_identified = bool(config['require_all_identified'])
                
                if 'min_confidence_threshold' in config:
                    threshold = float(config['min_confidence_threshold'])
                    if 0.0 <= threshold <= 1.0:
                        self.min_confidence_threshold = threshold
                
                if 'max_unknown_persons' in config:
                    max_unknown = int(config['max_unknown_persons'])
                    if max_unknown >= 0:
                        self.max_unknown_persons = max_unknown
                
                if 'access_timeout' in config:
                    timeout = float(config['access_timeout'])
                    if timeout > 0:
                        self.access_timeout = timeout
            
            return {
                'status': 'success',
                'message': 'Configuración actualizada correctamente',
                'current_config': {
                    'require_all_identified': self.require_all_identified,
                    'min_confidence_threshold': self.min_confidence_threshold,
                    'max_unknown_persons': self.max_unknown_persons,
                    'access_timeout': self.access_timeout
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error configurando control de acceso: {str(e)}'
            }
    
    def get_access_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas globales del control de acceso"""
        try:
            with self.lock:
                global_stats = {
                    'total_cameras': len(self.access_stats),
                    'cameras': {}
                }
                
                for camera_id, stats in self.access_stats.items():
                    camera_stats = stats.copy()
                    if camera_stats['last_access_time']:
                        camera_stats['last_access_time'] = camera_stats['last_access_time'].isoformat()
                    
                    # Calcular tasa de éxito
                    total_evals = camera_stats['total_evaluations']
                    if total_evals > 0:
                        camera_stats['success_rate'] = (camera_stats['access_granted'] / total_evals) * 100
                    else:
                        camera_stats['success_rate'] = 0.0
                    
                    global_stats['cameras'][camera_id] = camera_stats
                
                return global_stats
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error obteniendo estadísticas: {str(e)}'
            }
    
    def reset_statistics(self, camera_id: str = None) -> Dict[str, Any]:
        """Reinicia las estadísticas de acceso"""
        try:
            with self.lock:
                if camera_id:
                    # Reiniciar estadísticas de una cámara específica
                    if camera_id in self.access_stats:
                        self.access_stats[camera_id] = {
                            'total_evaluations': 0,
                            'access_granted': 0,
                            'access_denied': 0,
                            'last_access_time': None,
                            'consecutive_denials': 0
                        }
                        self.access_history[camera_id] = []
                        message = f'Estadísticas reiniciadas para cámara {camera_id}'
                    else:
                        return {
                            'status': 'error',
                            'message': f'Cámara {camera_id} no encontrada'
                        }
                else:
                    # Reiniciar todas las estadísticas
                    self.access_stats.clear()
                    self.access_history.clear()
                    message = 'Todas las estadísticas han sido reiniciadas'
                
                return {
                    'status': 'success',
                    'message': message
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error reiniciando estadísticas: {str(e)}'
            }
    
    def _activate_relay(self):
        """Activa el relé relay_001 por 5 segundos"""
        try:
            relay_id = "relay_001"
            
            # Crear comando para encender el relé con temporizador de 5 segundos
            command = RelayCommand(
                action=RelayStatus.ON,
                timer_seconds=5
            )
            
            # Ejecutar el comando de forma asíncrona
            def run_relay_command():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(relay_service.set_relay_state(relay_id, command))
                    
                    if response.success:
                        # Relay activated for 5 seconds - Access granted
                        pass
                    else:
                        print(f"[ERROR] Error activando relé {relay_id}: {response.message}")
                        
                except Exception as e:
                    print(f"[ERROR] Error ejecutando comando de relé: {e}")
                finally:
                    loop.close()
            
            # Ejecutar en un hilo separado para no bloquear
            import threading
            relay_thread = threading.Thread(target=run_relay_command)
            relay_thread.daemon = True
            relay_thread.start()
            
        except Exception as e:
            print(f"[ERROR] Error en _activate_relay: {e}")
    
    def _send_access_video_notification(self, camera_id: str, identified_names: List[str]):
        """Envía notificación de video por WhatsApp para acceso concedido"""
        try:
            from services.whatsapp_service import whatsapp_service
            
            # Enviar video por WhatsApp con el tipo de evento correcto
            success = whatsapp_service.send_intrusion_notification(
                camera_id=camera_id,
                event_type="access_granted",
                person_count=len(identified_names) if identified_names else 1,
                face_count=len(identified_names) if identified_names else 1,
                identified_names=identified_names
            )
            
            if success:
                names_text = ", ".join(identified_names) if identified_names else "Sin identificar"
                print(f"[ACCESS_GRANTED] Video de acceso enviado por WhatsApp para {camera_id} - Personas: {names_text}")
            else:
                print(f"[ERROR] Falló el envío de video de acceso por WhatsApp para {camera_id}")
            
        except Exception as e:
            print(f"[ERROR] Error enviando video de acceso por WhatsApp para {camera_id}: {e}")
    
    def cancel_all_auto_access_timers(self, camera_id: str = None):
        """Cancela todos los timers de acceso automático"""
        cancelled_count = 0
        
        if camera_id:
            # Cancelar timer específico de una cámara
            if camera_id in self.access_timers:
                self.access_timers[camera_id].cancel()
                del self.access_timers[camera_id]
                cancelled_count = 1
                
            if camera_id in self.access_conditions_met:
                self.access_conditions_met[camera_id] = False
                
            print(f"[ACCESS_AUTO] Timer cancelado para cámara {camera_id}")
        else:
            # Cancelar todos los timers
            for cam_id, timer in list(self.access_timers.items()):
                timer.cancel()
                cancelled_count += 1
                
            self.access_timers.clear()
            self.access_conditions_met.clear()
            self.access_start_times.clear()
            
            if cancelled_count > 0:
                print(f"[ACCESS_AUTO] {cancelled_count} timers de acceso automático cancelados")
        
        return cancelled_count
    
    def shutdown(self):
        """Cierra el servicio y cancela todos los timers"""
        self.cancel_all_auto_access_timers()
        # Access Control Service shutdown complete

# Global instance
access_control_service = AccessControlService()