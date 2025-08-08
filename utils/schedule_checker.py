from datetime import datetime, time
import pytz
from typing import Dict, Any

class NotificationScheduleChecker:
    """
    Clase para verificar si las notificaciones de WhatsApp deben enviarse
    basándose en los horarios configurados para cada cámara.
    """
    
    def __init__(self):
        self.colombia_tz = pytz.timezone('America/Bogota')
    
    def is_notification_time_allowed(self, camera_config: Dict[str, Any]) -> bool:
        """
        Verifica si la hora actual está dentro del rango permitido para enviar notificaciones.
        
        Args:
            camera_config: Configuración de la cámara que incluye notification_schedule
            
        Returns:
            bool: True si se pueden enviar notificaciones, False en caso contrario
        """
        # Si no hay configuración de horarios o está deshabilitada, permitir siempre
        schedule_config = camera_config.get('notification_schedule', {})
        if not schedule_config.get('enabled', False):
            return True
        
        # Obtener la hora actual en Colombia
        now_colombia = datetime.now(self.colombia_tz)
        current_time = now_colombia.time()
        
        # Obtener horarios configurados
        start_time_str = schedule_config.get('start_time', '18:00')
        end_time_str = schedule_config.get('end_time', '08:00')
        
        # Convertir strings a objetos time
        start_time = self._parse_time_string(start_time_str)
        end_time = self._parse_time_string(end_time_str)
        
        # Verificar si el rango cruza medianoche (ej: 18:00 - 08:00)
        if start_time > end_time:
            # El rango cruza medianoche
            # Permitir si la hora actual es >= start_time O <= end_time
            return current_time >= start_time or current_time <= end_time
        else:
            # El rango no cruza medianoche (ej: 08:00 - 18:00)
            # Permitir si la hora actual está entre start_time y end_time
            return start_time <= current_time <= end_time
    
    def _parse_time_string(self, time_str: str) -> time:
        """
        Convierte un string de tiempo en formato HH:MM a un objeto time.
        
        Args:
            time_str: String en formato "HH:MM"
            
        Returns:
            time: Objeto time correspondiente
        """
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour, minute)
        except (ValueError, AttributeError):
            # En caso de error, retornar medianoche
            return time(0, 0)
    
    def get_current_colombia_time(self) -> datetime:
        """
        Obtiene la hora actual en Colombia.
        
        Returns:
            datetime: Hora actual en zona horaria de Colombia
        """
        return datetime.now(self.colombia_tz)
    
    def get_schedule_info(self, camera_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtiene información detallada sobre el horario de notificaciones.
        
        Args:
            camera_config: Configuración de la cámara
            
        Returns:
            dict: Información del horario incluyendo estado actual
        """
        schedule_config = camera_config.get('notification_schedule', {})
        current_time = self.get_current_colombia_time()
        is_allowed = self.is_notification_time_allowed(camera_config)
        
        return {
            'enabled': schedule_config.get('enabled', False),
            'start_time': schedule_config.get('start_time', '18:00'),
            'end_time': schedule_config.get('end_time', '08:00'),
            'timezone': schedule_config.get('timezone', 'America/Bogota'),
            'current_time': current_time.strftime('%H:%M:%S'),
            'current_date': current_time.strftime('%Y-%m-%d'),
            'notifications_allowed': is_allowed
        }

# Instancia global del verificador de horarios
schedule_checker = NotificationScheduleChecker()