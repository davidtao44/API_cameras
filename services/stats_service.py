from datetime import datetime
from app.config.settings import CAMERAS

# Diccionario para almacenar estadísticas por cámara
camera_stats = {
    cam_id: {
        "total_detections": 0,
        "recognized": 0,
        "unrecognized": 0,
        "recognized_names": set(),
        "last_updated": datetime.now()
    }
    for cam_id in CAMERAS
}

def update_stats(camera_id: str, name: str):
    """Actualiza las estadísticas de reconocimiento"""
    stats = camera_stats[camera_id]
    stats["total_detections"] += 1
    if name != "Desconocido":
        stats["recognized"] += 1
        stats["recognized_names"].add(name)
    else:
        stats["unrecognized"] += 1
    stats["last_updated"] = datetime.now()

def get_camera_stats(camera_id: str):
    """Obtiene las estadísticas de una cámara específica"""
    if camera_id not in camera_stats:
        return None
    
    stats = camera_stats[camera_id]
    total = stats["total_detections"] or 1
    
    return {
        "total_detections": stats["total_detections"],
        "recognized": stats["recognized"],
        "unrecognized": stats["unrecognized"],
        "recognition_rate": stats["recognized"] / total,
        "last_updated": stats["last_updated"],
        "recognized_names": list(stats["recognized_names"])
    }