import cv2
import time
from services.stats_service import update_stats
from services.camera_stream import stream_frames_with_digest, stream_frames_without_auth, stream_recognizer
from surveillance.enhanced_frame_processing_service import enhanced_frame_processing_service

def generate_stream(config, camera_id: str):
    url = config["url"]
    
    # Start frame processing for this camera if not already started
    enhanced_frame_processing_service.start_processing_for_camera(camera_id)
    
    # Control de tiempo para análisis de frames
    last_analysis_time = 0
    analysis_interval = 0.5  # 500ms entre análisis
    last_detected_names = []  # Mantener últimos nombres detectados
    last_person_count = 0  # Mantener último conteo de personas
    
    if config.get("auth_required", True):
        user = config["username"]
        pwd = config["password"]
        frame_generator = stream_frames_with_digest(url, user, pwd)
    else:
        frame_generator = stream_frames_without_auth(url)
    
    for frame in frame_generator:
        current_time = time.time()
        
        # REMOVED: Frame submission moved to surveillance_service.py to centralize control
        # enhanced_frame_processing_service.submit_frame_for_processing(frame, camera_id)
        
        # Get latest processing result
        processing_result = enhanced_frame_processing_service.get_latest_processing_result(camera_id)
        
        if processing_result:
            # Always use processed frame from centralized service
            analyzed_frame = processing_result.processed_frame
            
            # Update cached data if enough time has passed
            if current_time - last_analysis_time >= analysis_interval:
                last_detected_names = processing_result.detected_names
                last_person_count = processing_result.person_count
                last_analysis_time = current_time
                
                # Actualizar estadísticas solo cuando se detectan nombres
                for name in last_detected_names:
                    if name != "Unknown":
                        update_stats(camera_id, name)
        else:
            # Create a basic processed frame while waiting for processing results
            analyzed_frame = create_waiting_frame(frame, camera_id)
        
        ret, jpeg = cv2.imencode('.jpg', analyzed_frame)
        if not ret:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        )

def create_waiting_frame(frame, camera_id):
    """Create a basic processed frame while waiting for processing results"""
    processed_frame = frame.copy()
    
    # Add processing status
    cv2.putText(processed_frame, "Processing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Add camera ID
    cv2.putText(processed_frame, f"Camera: {camera_id}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw ROI if configured
    processed_frame = stream_recognizer._draw_roi_on_frame(processed_frame, camera_id)
    
    return processed_frame

def draw_previous_detections(frame, names, person_count):
    """Dibuja un indicador simple de las últimas detecciones sin procesar el frame"""
    # Mostrar contador de personas detectadas en la esquina
    text = f"Personas detectadas: {person_count}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if names:
        # Mostrar nombres en la parte inferior
        y_offset = frame.shape[0] - 60
        for i, name in enumerate(names[:3]):  # Máximo 3 nombres
            cv2.putText(frame, name, (10, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame