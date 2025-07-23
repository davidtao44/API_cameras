CAMERAS = {
    "cam1": {
        "url": "http://172.16.2.126:8080/NxjRXXU0vFm9fklpQqDbmbz4LBxcnq/mjpeg/wcWasVvhGm/79YVnwoDVg",
        "username": "admin",
        "password": "Tecon2025#",
        "auth_required": False,
        "access_control_enabled": True,  # Control de acceso y envio de mensajes por http
        "alert_control_enabled": True,  # Control de alarma y bloqueo de peticiones al relé
        "whatsapp_notifications_enabled": True,  # Envío de alertas por WhatsApp
        "notification_phones": ["573107834112","573222262665"],  # Números específicos por cámara
        "roi": {
            "enabled": True,        # ROI habilitada/deshabilitada
            "x": 0.0,               # Coordenada X de inicio (0.0-1.0 para porcentajes)
            "y": 0.0,               # Coordenada Y de inicio (0.0-1.0 para porcentajes)
            "width": 1.0,           # Ancho de la ROI (0.0-1.0 para porcentajes)
            "height": 1.0,          # Alto de la ROI (0.0-1.0 para porcentajes)
            "use_percentage": True, # Usar porcentajes (True) o píxeles (False)
            "rotation": 0,          # Rotación en grados (0, 90, 180, 270)
            "shape": "polygon",   # Forma: "rectangle", "circle", "polygon"
            "polygon_points": [[0.5, 0],   # Punto 1
                               [0.5, 1],   # Punto 2
                               [1, 1],   # Punto 3
                               [1, 0]],   # Puntos para ROI poligonal [(x1,y1), (x2,y2), ...]
            "adaptive": False,      # ROI adaptativa basada en detecciones previas
            "margin": 0.1          # Margen adicional para ROI adaptativa
        }
    },
    "cam2": {
        "url": "http://172.16.2.126:8080/NxjRXXU0vFm9fklpQqDbmbz4LBxcnq/mjpeg/wcWasVvhGm/A5vjFix0zS",
        "username": "admin",
        "password": "Tecon2025#",
        "auth_required": False,
        "access_control_enabled": False,  # Control de acceso habilitado
        "alert_control_enabled": False,  # Control de alarma y bloqueo de peticiones al relé
        "whatsapp_notifications_enabled": False,  # Envío de alertas por WhatsApp
        "notification_phones": ["573107834112", "573132843357", "573209838287" , "573228097407"],  # Solo un número para esta cámara
        "roi": {
            "enabled": True,        # ROI habilitada/deshabilitada
            "x": 0.0,               # Coordenada X de inicio (0.0-1.0 para porcentajes)
            "y": 0.0,               # Coordenada Y de inicio (0.0-1.0 para porcentajes)
            "width": 1.0,           # Ancho de la ROI (0.0-1.0 para porcentajes)
            "height": 1.0,          # Alto de la ROI (0.0-1.0 para porcentajes)
            "use_percentage": True, # Usar porcentajes (True) o píxeles (False)
            "rotation": 0,          # Rotación en grados (0, 90, 180, 270)
            "shape": "polygon",   # Forma: "rectangle", "circle", "polygon"
            "polygon_points": [[0, 0.1],   # Punto 1
                               [0, 1],   # Punto 2
                               [1, 1],   # Punto 3
                               [1, 0.7]],   # Puntos para ROI poligonal [(x1,y1), (x2,y2), ...]
            "adaptive": False,      # ROI adaptativa basada en detecciones previas
            "margin": 0.1          # Margen adicional para ROI adaptativa
        }
    },
    "cam3": {
        "url": "http://172.16.2.126:8080/NxjRXXU0vFm9fklpQqDbmbz4LBxcnq/mjpeg/wcWasVvhGm/lKHQOgspnC",
        "username": "admin",
        "password": "Tecon2025#",
        "auth_required": False,
        "access_control_enabled": False,  # Control de acceso habilitado
        "alert_control_enabled": False,  # Control de alarma y bloqueo de peticiones al relé
        "whatsapp_notifications_enabled": False,  # Envío de alertas por WhatsApp
        "notification_phones": ["573107834112", "573132843357", "573209838287" , "573228097407"],  # Solo un número para esta cámara
        "roi": {
            "enabled": False,
            "x": 0.0,
            "y": 0.0,
            "width": 1.0,
            "height": 1.0,
            "use_percentage": True,
            "rotation": 0,
            "shape": "rectangle",
            "polygon_points": [],
            "adaptive": False,
            "margin": 0.1
        }
    }
}