import cv2
import time
from services.stats_service import update_stats
from services.camera_stream import stream_frames_with_digest, stream_frames_without_auth, stream_recognizer

def generate_stream(config, camera_id: str):
    url = config["url"]
    
    # Control de tiempo para análisis de frames
    last_analysis_time = 0
    analysis_interval = 0.01  # 1 segundo entre análisis
    last_detected_names = []  # Mantener últimos nombres detectados
    
    if config.get("auth_required", True):
        user = config["username"]
        pwd = config["password"]
        frame_generator = stream_frames_with_digest(url, user, pwd)
    else:
        frame_generator = stream_frames_without_auth(url)
    
    for frame in frame_generator:
        current_time = time.time()
        
        # Solo analizar cada segundo
        if current_time - last_analysis_time >= analysis_interval:
            # Realizar análisis de reconocimiento facial
            analyzed_frame, names = stream_recognizer.recognize_for_stream(frame)
            last_detected_names = names
            last_analysis_time = current_time
            
            # Actualizar estadísticas solo cuando se detectan nombres
            for name in names:
                update_stats(camera_id, name)
        else:
            # Solo dibujar las detecciones anteriores sin procesar
            analyzed_frame = draw_previous_detections(frame, last_detected_names)
        
        ret, jpeg = cv2.imencode('.jpg', analyzed_frame)
        if not ret:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        )

def draw_previous_detections(frame, names):
    """Dibuja un indicador simple de las últimas detecciones sin procesar el frame"""
    if names:
        # Mostrar contador de personas detectadas en la esquina
        text = f"Personas detectadas: {len(names)}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar nombres en la parte inferior
        y_offset = frame.shape[0] - 60
        for i, name in enumerate(names[:3]):  # Máximo 3 nombres
            cv2.putText(frame, name, (10, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame