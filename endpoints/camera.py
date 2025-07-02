from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from services.camera_service import generate_stream
from config.settings import CAMERAS

router = APIRouter(prefix="/stream", tags=["camera"])

@router.get("/{camera_id}")
def video_feed(camera_id: str):
    if camera_id not in CAMERAS:
        return Response(content="Cámara no encontrada", status_code=404)
    return StreamingResponse(
        generate_stream(CAMERAS[camera_id], camera_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )