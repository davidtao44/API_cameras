CAMERAS = {
    "cam1": {
        "url": "http://172.16.2.231:8080/NxjRXXU0vFm9fklpQqDbmbz4LBxcnq/mjpeg/wcWasVvhGm/79YVnwoDVg",
        "username": "admin",
        "password": "Tecon2025#",
        "auth_required": False,
        "access_control_enabled": False,  # Control de acceso y envio de mensajes por http
        "http_notifications_enabled": False,  # Control de alamar y envio de imagenes por http
        "notification_phones": ["573107834112"]  # Números específicos por cámara
    },
    "cam2": {
        "url": "http://172.16.2.231:8080/NxjRXXU0vFm9fklpQqDbmbz4LBxcnq/mjpeg/wcWasVvhGm/A5vjFix0zS",
        "username": "admin",
        "password": "Tecon2025#",
        "auth_required": False,
        "access_control_enabled": True,  # Control de acceso habilitado
        "http_notifications_enabled": True,  # Notificaciones habilitadas
        "notification_phones": ["573107834112"]  # Solo un número para esta cámara
    }
}