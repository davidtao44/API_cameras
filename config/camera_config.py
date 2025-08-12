CAMERAS = {
    "cam1": {
        "url": "http://172.16.2.231:8080/NxjRXXU0vFm9fklpQqDbmbz4LBxcnq/mjpeg/wcWasVvhGm/79YVnwoDVg",
        "username": "admin",
        "password": "Tecon2025#",
        "auth_required": False,
        "access_control_enabled": True,  # Control de acceso y envio de mensajes por http
        "alert_control_enabled": True,  # Control de alarma y bloqueo de peticiones al relé
        "whatsapp_notifications_enabled": True,  # Envío de alertas por WhatsApp
        "notification_phones": ['573107834112', '573222262665', '573228097407'],# '573132843357', '573209838287', '573228097407'],  # Números específicos por cámara
        "video_duration_seconds": 20,
        "notification_schedule": {
            "enabled": False,  # Habilitar control de horarios
            "start_time": "17:30",  # Hora de inicio (6:00 PM)
            "end_time": "08:00",    # Hora de fin (8:00 AM del día siguiente)
            "timezone": "America/Bogota"  # Zona horaria de Colombia
        },
        "roi": {
            "enabled": False,        # ROI habilitada/deshabilitada
            "x": 0.0,               # Coordenada X de inicio (0.0-1.0 para porcentajes)
            "y": 0.0,               # Coordenada Y de inicio (0.0-1.0 para porcentajes)
            "width": 0.2,           # Ancho de la ROI (0.0-1.0 para porcentajes)
            "height": 1,          # Alto de la ROI (0.0-1.0 para porcentajes)
            "use_percentage": True, # Usar porcentajes (True) o píxeles (False)
            "rotation": 0,          # Rotación en grados (0, 90, 180, 270)
            "shape": "rectangle",   # Forma: "rectangle", "circle", "polygon"
            "polygon_points": [[0.0, 0], [0.0, 1], [1, 1], [1, 0]],   # Puntos para ROI poligonal [(x1,y1), (x2,y2), ...]
            "adaptive": False,      # ROI adaptativa basada en detecciones previas
            "margin": 0.1,          # Margen adicional para ROI adaptativa
        },
        "detection_config": {
            "min_face_size": 60,           # Tamaño mínimo de rostro para detección (píxeles)
            "max_distance_threshold": 200, # Distancia máxima para asociar rostros con personas (píxeles)
            "person_confidence": 0.9,      # Confianza mínima para detección de personas (0.0-1.0)
            "face_confidence": 0.35,        # Confianza mínima para reconocimiento facial (0.0-1.0)
        },
    },
    "cam2": {
        "url": "http://172.16.2.231:8080/NxjRXXU0vFm9fklpQqDbmbz4LBxcnq/mjpeg/wcWasVvhGm/A5vjFix0zS",
        "username": "admin",
        "password": "Tecon2025#",
        "auth_required": False,
        "access_control_enabled": True,  # Control de acceso y envio de mensajes por http
        "alert_control_enabled": True,  # Control de alarma y bloqueo de peticiones al relé
        "whatsapp_notifications_enabled": True,  # Envío de alertas por WhatsApp
        "notification_phones": ['573107834112', '573132843357', '573209838287', '573228097407'],  # Números específicos por cámara
        "video_duration_seconds": 10,
        "notification_schedule": {
            "enabled": False,  # Habilitar control de horarios
            "start_time": "16:00",  # Hora de inicio (10:00 PM)
            "end_time": "06:00",    # Hora de fin (6:00 AM del día siguiente)
            "timezone": "America/Bogota"  # Zona horaria de Colombia
        },
        "roi": {
            "enabled": True,        # ROI habilitada/deshabilitada
            "x": 0.0,               # Coordenada X de inicio (0.0-1.0 para porcentajes)
            "y": 0.0,               # Coordenada Y de inicio (0.0-1.0 para porcentajes)
            "width": 1.0,           # Ancho de la ROI (0.0-1.0 para porcentajes)
            "height": 1.0,          # Alto de la ROI (0.0-1.0 para porcentajes)
            "use_percentage": True, # Usar porcentajes (True) o píxeles (False)
            "rotation": 0,          # Rotación en grados (0, 90, 180, 270)
            "shape": "polygon",   # Forma: "rectangle", "circle", "polygon"
            "polygon_points": [[0, 0.1], [0, 1], [1, 1], [1, 0.7]],   # Puntos para ROI poligonal [(x1,y1), (x2,y2), ...]
            "adaptive": False,      # ROI adaptativa basada en detecciones previas
            "margin": 0.1,          # Margen adicional para ROI adaptativa
        },
        "detection_config": {
            "min_face_size": 60,           # Tamaño mínimo de rostro para detección (píxeles)
            "max_distance_threshold": 300, # Distancia máxima para asociar rostros con personas (píxeles)
            "person_confidence": 0.4,      # Confianza mínima para detección de personas (0.0-1.0)
            "face_confidence": 0.35,        # Confianza mínima para reconocimiento facial (0.0-1.0)
        },
    },
    "cam3": {
        "url": "http://172.16.2.231:8080/NxjRXXU0vFm9fklpQqDbmbz4LBxcnq/mjpeg/wcWasVvhGm/lKHQOgspnC",
        "username": "admin",
        "password": "Tecon2025#",
        "auth_required": False,
        "access_control_enabled": True,  # Control de acceso y envio de mensajes por http
        "alert_control_enabled": True,  # Control de alarma y bloqueo de peticiones al relé
        "whatsapp_notifications_enabled": True,  # Envío de alertas por WhatsApp
        "notification_phones": ['573107834112', '573132843357', '573209838287', '573228097407'],  # Números específicos por cámara
        "video_duration_seconds": 10,
        "notification_schedule": {
            "enabled": False,  # Habilitar control de horarios (deshabilitado para cam3)
            "start_time": "20:00",  # Hora de inicio (8:00 PM)
            "end_time": "07:00",    # Hora de fin (7:00 AM del día siguiente)
            "timezone": "America/Bogota"  # Zona horaria de Colombia
        },
        "roi": {
            "enabled": False,        # ROI habilitada/deshabilitada
            "x": 0.0,               # Coordenada X de inicio (0.0-1.0 para porcentajes)
            "y": 0.0,               # Coordenada Y de inicio (0.0-1.0 para porcentajes)
            "width": 1.0,           # Ancho de la ROI (0.0-1.0 para porcentajes)
            "height": 1.0,          # Alto de la ROI (0.0-1.0 para porcentajes)
            "use_percentage": True, # Usar porcentajes (True) o píxeles (False)
            "rotation": 0,          # Rotación en grados (0, 90, 180, 270)
            "shape": "rectangle",   # Forma: "rectangle", "circle", "polygon"
            "adaptive": False,      # ROI adaptativa basada en detecciones previas
            "margin": 0.1,          # Margen adicional para ROI adaptativa
        },
        "detection_config": {
            "min_face_size": 70,           # Tamaño mínimo de rostro para detección (píxeles)
            "max_distance_threshold": 300, # Distancia máxima para asociar rostros con personas (píxeles)
            "person_confidence": 0.6,      # Confianza mínima para detección de personas (0.0-1.0)
            "face_confidence": 0.35,        # Confianza mínima para reconocimiento facial (0.0-1.0)
        },
    },
}
