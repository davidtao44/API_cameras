from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import time
from config.firebase import initialize_firebase
from endpoints import camera, stats, access, relay, access_control
from services.firebase_service import monitor_firebase_visitors
from services.surveillance_service import surveillance_service  # ✅ Agregar import

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
app.include_router(access.router)
app.include_router(relay.router)
app.include_router(access_control.router)  # ✅ Agregar el nuevo router

# Inicializar Firebase
firebase_initialized = initialize_firebase()

@app.get("/")
def root():
    return {"message": "API cámaras funcionando!"}

if __name__ == "__main__":
    # Iniciar el hilo de monitoreo de Firebase
    firebase_monitor_thread = threading.Thread(
        target=monitor_firebase_visitors, 
        daemon=True,
        kwargs={"firebase_initialized": firebase_initialized}
    )
    firebase_monitor_thread.start()
    
    # ✅ Iniciar el monitoreo automático de surveillance para cam1
    surveillance_thread = threading.Thread(
        target=surveillance_service.start_monitoring,
        args=("cam2",),
        daemon=True,
        name="surveillance_cam1"
    )
    surveillance_thread.start()
    print("🎥 Iniciando surveillance automático para cam1")
    
    # Iniciar el servidor FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)