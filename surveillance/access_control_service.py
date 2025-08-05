import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time

# Importar servicios necesarios
from .enhanced_frame_processing_service import enhanced_frame_processing_service, EnhancedFrameProcessingResult
from config.settings import CAMERAS
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
        
        # Estadísticas
        self.access_stats = defaultdict(lambda: {
            'total_evaluations': 0,
            'access_granted': 0,
            'access_denied': 0,
            'last_access_time': None,
            'consecutive_denials': 0
        })
        
        print("[INFO] Access Control Service initialized")
    
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
                print(f"[INFO] Control de acceso deshabilitado para cámara {camera_id}")
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
            
            # Si se concede el acceso, activar el relé
            if access_result.access_granted:
                self._activate_relay()
            
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
    
    def _evaluate_access_internal(self, processing_result: EnhancedFrameProcessingResult) -> AccessControlResult:
        """Lógica interna de evaluación de acceso"""
        camera_id = processing_result.camera_id
        timestamp = processing_result.timestamp
        
        # Obtener información de personas detectadas
        total_persons = processing_result.person_count
        face_results = processing_result.face_results
        
        # Condición 1: Debe haber más de una persona
        if total_persons <= 1:
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
        
        identified_persons = len(identified_names)
        unknown_persons = unknown_count
        
        # Calcular nivel de confianza promedio
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Condición 2: El número de rostros debe ser mayor que el número de personas
        total_faces = len(face_results)
        if total_faces <= total_persons:
            return AccessControlResult(
                camera_id=camera_id,
                access_granted=False,
                total_persons=total_persons,
                identified_persons=identified_persons,
                unknown_persons=unknown_persons,
                identified_names=identified_names,
                unknown_count=unknown_count,
                timestamp=timestamp,
                confidence_level=avg_confidence
            )
        
        # Condición 3: Todos los rostros deben estar identificados (no ser "desconocido")
        access_granted = unknown_count == 0 and identified_persons > 0
        
        return AccessControlResult(
            camera_id=camera_id,
            access_granted=access_granted,
            total_persons=total_persons,
            identified_persons=identified_persons,
            unknown_persons=unknown_persons,
            identified_names=identified_names,
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
                        print(f"[INFO] Relé {relay_id} activado por 5 segundos - Acceso concedido")
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

# Global instance
access_control_service = AccessControlService()