from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from camera_stream import stream_frames_with_digest, stream_frames_without_auth, recognizer
import cv2
import uvicorn
from typing import Dict, List
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

# Modelo para la respuesta de estadísticas
class RecognitionStats(BaseModel):
    total_detections: int
    recognized: int
    unrecognized: int
    recognition_rate: float
    last_updated: datetime
    recognized_names: List[str]

CAMERAS = {
    "cam1": {
        "url": "http://172.16.2.221/axis-cgi/mjpg/video.cgi",
        "username": "root",
        "password": "admin",
        "auth_required": True
    },
    # Ejemplo de cámara sin autenticación
    "cam2": {
        "url": "http://172.16.2.231:8080/NxjRXXU0vFm9fklpQqDbmbz4LBxcnq/mjpeg/wcWasVvhGm/Vz3RhZJXBQ",
        "auth_required": False
    }
}

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

def generate_stream(config, camera_id: str):
    url = config["url"]
    
    if config.get("auth_required", True):
        # Flujo con autenticación
        user = config["username"]
        pwd = config["password"]
        frame_generator = stream_frames_with_digest(url, user, pwd)
    else:
        # Flujo sin autenticación
        frame_generator = stream_frames_without_auth(url)
    
    for frame in frame_generator:
        # APLICAR RECONOCIMIENTO FACIAL
        frame, names = recognizer.recognize(frame)
        
        # Actualizar estadísticas por cada cara detectada
        for name in names:
            update_stats(camera_id, name)
        
        # ENCODE PARA STREAMING
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        )

@app.get("/stream/{camera_id}")
def video_feed(camera_id: str):
    if camera_id not in CAMERAS:
        return Response(content="Cámara no encontrada", status_code=404)
    return StreamingResponse(generate_stream(CAMERAS[camera_id], camera_id),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/stats/{camera_id}", response_model=RecognitionStats)
def get_stats(camera_id: str):
    """Endpoint para obtener estadísticas de reconocimiento"""
    if camera_id not in camera_stats:
        return Response(content="Cámara no encontrada", status_code=404)
    
    stats = camera_stats[camera_id]
    total = stats["total_detections"] or 1  # Evitar división por cero
    
    return {
        "total_detections": stats["total_detections"],
        "recognized": stats["recognized"],
        "unrecognized": stats["unrecognized"],
        "recognition_rate": stats["recognized"] / total,
        "last_updated": stats["last_updated"],
        "recognized_names": list(stats["recognized_names"])
    }

@app.get("/people-count/{camera_id}")
def get_people_count(camera_id: str):
    """Endpoint para obtener el conteo de personas actual"""
    if camera_id not in camera_stats:
        return Response(content="Cámara no encontrada", status_code=404)
    
    stats = camera_stats[camera_id]
    return {
        "current_people": stats["recognized"] + stats["unrecognized"],
        "recognized": stats["recognized"],
        "unrecognized": stats["unrecognized"]
    }

@app.get("/recognized-people/{camera_id}")
def get_recognized_people(camera_id: str):
    """Endpoint para obtener la lista de personas reconocidas"""
    if camera_id not in camera_stats:
        return Response(content="Cámara no encontrada", status_code=404)
    
    return {
        "recognized_people": list(camera_stats[camera_id]["recognized_names"]),
        "count": len(camera_stats[camera_id]["recognized_names"])
    }

@app.get("/")
def root():
    return Response(content="API cámaras funcionando!", status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)