CAMERAS = {
    "cam1": {
<<<<<<< HEAD
        "url": "http://172.16.2.221/axis-cgi/mjpg/video.cgi",
        "username": "root",
        "password": "admin",
        "auth_required": True
    },
    "cam2": {
        "url": "http://172.16.2.17/video1s1.mjpg",
        "username": "root",
        "password": "Tecon2025",
        "auth_required": True
    },

    "cam3": {
        "url": "http://172.16.2.17/video1s3.mjpg",
        "username": "root",
        "password": "Tecon2025",
        "auth_required": True
    },
    "cam4": {
        "url": "http://172.16.2.221/axis-cgi/mjpg/video.cgi?camera=1",
        "username": "root",
        "password": "admin",
        "auth_required": True
    }, 
    "cam5": {
        "url": "http://172.16.2.221/axis-cgi/mjpg/video.cgi?camera=2",
        "username": "root",
        "password": "admin",
        "auth_required": True
    },
    "cam6": {
        "url": "http://172.16.2.221/axis-cgi/mjpg/video.cgi?camera=3",
        "username": "root",
        "password": "admin",
        "auth_required": True
    },
    "cam7": {
        "url": "http://172.16.2.221/axis-cgi/mjpg/video.cgi?camera=4",
        "username": "root",
        "password": "admin",
        "auth_required": True
    },

    "cam8": {
=======
>>>>>>> feature/human_detection
        "url": "http://172.16.2.231:8080/NxjRXXU0vFm9fklpQqDbmbz4LBxcnq/mjpeg/wcWasVvhGm/79YVnwoDVg",
        "username": "admin",
        "password": "Tecon2025#",
        "auth_required": False,
        "access_control_enabled": False,  # Control de acceso y envio de mensajes por http
        "http_notifications_enabled": False,  # Control de alamar y envio de imagenes por http
        "notification_phones": ["573107834112", "573132843357"]  # Números específicos por cámara
    },
    "cam2": {
        "url": "http://172.16.2.231:8080/NxjRXXU0vFm9fklpQqDbmbz4LBxcnq/mjpeg/wcWasVvhGm/A5vjFix0zS",
        "username": "admin",
        "password": "Tecon2025#",
        "auth_required": False,
        "access_control_enabled": True,  # Control de acceso habilitado
        "http_notifications_enabled": True,  # Notificaciones habilitadas
        "notification_phones": ["573107834112", "573132843357"]  # Solo un número para esta cámara
    },
    "cam3": {
        "url": "http://172.16.2.231:8080/NxjRXXU0vFm9fklpQqDbmbz4LBxcnq/mjpeg/wcWasVvhGm/lKHQOgspnC",
        "username": "admin",
        "password": "Tecon2025#",
        "auth_required": False,
        "access_control_enabled": False,  # Control de acceso habilitado
        "http_notifications_enabled": False,  # Notificaciones habilitadas
        "notification_phones": ["573107834112", "573132843357"]  # Solo un número para esta cámara
    }
}