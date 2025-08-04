from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import time
from config.firebase import initialize_firebase
from endpoints import camera, stats, config, whatsapp
from services.firebase_service import monitor_firebase_visitors
from surveillance.surveillance_service import surveillance_service



app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(camera.router)
app.include_router(stats.router)
app.include_router(config.router)
app.include_router(whatsapp.router)

# Inicializar Firebase
firebase_initialized = initialize_firebase()

@app.get("/")
def root():
    return {
        "message": "API de detección de personas funcionando!",
        "status": "Sistema simplificado - Solo detección de personas conocidas/desconocidas"
    }

if __name__ == "__main__":
    # Iniciar el hilo de monitoreo de Firebase
    firebase_monitor_thread = threading.Thread(
        target=monitor_firebase_visitors, 
        daemon=True,
        kwargs={"firebase_initialized": firebase_initialized}
    )
    firebase_monitor_thread.start()
    
    surveillance_thread1 = threading.Thread(
        target=lambda: surveillance_service.start_monitoring("cam1"),
        daemon=True,
        name="surveillance_cam2"
    )
    surveillance_thread1.start()
    print("[INFO] Iniciando surveillance automático para cam1")

    surveillance_thread2 = threading.Thread(
        target=lambda: surveillance_service.start_monitoring("cam2"),
        daemon=True,
        name="surveillance_cam2"
    )
    surveillance_thread2.start()
    print("[INFO] Iniciando surveillance automático para cam2")

    surveillance_thread3 = threading.Thread(
        target=lambda: surveillance_service.start_monitoring("cam3"),
        daemon=True,
        name="surveillance_cam3"
    )
    surveillance_thread3.start()
    print("[INFO] Iniciando surveillance automático para cam3")
    
    # Iniciar el servidor FastAPI
    print("[INFO] Iniciando servidor FastAPI en puerto 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)