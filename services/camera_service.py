import cv2
from services.stats_service import update_stats
from services.camera_stream import stream_frames_with_digest, stream_frames_without_auth, stream_recognizer

def generate_stream(config, camera_id: str):
    url = config["url"]
    
    if config.get("auth_required", True):
        user = config["username"]
        pwd = config["password"]
        frame_generator = stream_frames_with_digest(url, user, pwd)
    else:
        frame_generator = stream_frames_without_auth(url)
    
    for frame in frame_generator:
        # Usar el recognizer simplificado solo para streaming
        frame, names = stream_recognizer.recognize_for_stream(frame)
        
        for name in names:
            update_stats(camera_id, name)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        )