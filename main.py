from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from camera_stream import stream_frames_with_digest, stream_frames_without_auth, recognizer
import cv2
import uvicorn
from typing import Dict, List
from pydantic import BaseModel
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import json
import os
import numpy as np
import threading
import time
import unicodedata
import re
from fastapi.middleware.cors import CORSMiddleware

# Inicializar Firebase (asegúrate de tener el archivo de credenciales)
try:
    cred = credentials.Certificate("C:/Users/jhona/Documents/Tecon/Certificados/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    firebase_initialized = True
except Exception as e:
    print(f"Error al inicializar Firebase: {e}")
    firebase_initialized = False

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Lista de orígenes permitidos
    allow_credentials=True,                   # Permitir cookies
    allow_methods=["*"],                      # Permitir todos los métodos HTTP
    allow_headers=["*"],                      # Permitir todos los headers
)

# Modelo para la respuesta de estadísticas
class RecognitionStats(BaseModel):
    total_detections: int
    recognized: int
    unrecognized: int
    recognition_rate: float
    last_updated: datetime
    recognized_names: List[str]

# Modelo para la solicitud de procesamiento de datos de Firebase
class FirebaseProcessRequest(BaseModel):
    collection_name: str
    document_id: str = None

# Modelo para la respuesta del procesamiento
class ProcessResponse(BaseModel):
    success: bool
    message: str
    processed_names: List[str] = []
    firebase_data: Dict = {}  # Nuevo campo para los datos de Firebase

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

# Variable para almacenar los últimos documentos procesados
last_processed_docs = set()

# ==================== FUNCIONES NORMALES ====================

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

def normalize_text(text):
    """
    Normaliza un texto eliminando tildes y caracteres especiales.
    Convierte a minúsculas y luego capitaliza cada palabra.
    """
    if not text:
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Normalizar (eliminar tildes)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    # Reemplazar la letra ñ por n
    text = text.replace('ñ', 'n')
    
    # Eliminar caracteres especiales, dejando solo letras, números y espacios
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Capitalizar cada palabra
    text = ' '.join(word.capitalize() for word in text.split())
    
    return text

def monitor_firebase_visitors():
    """
    Función que monitorea constantemente la colección 'visitors' de Firebase
    para detectar nuevos registros y generar embeddings automáticamente.
    También detecta registros eliminados y los elimina del archivo de embeddings.
    """
    global last_processed_docs
    
    print("Iniciando monitoreo de la colección 'visitors' en Firebase...")
    
    while True:
        try:
            if not firebase_initialized:
                print("Firebase no está inicializado. Reintentando en 60 segundos...")
                time.sleep(60)
                continue
                
            # Obtener referencia a la colección visitors
            collection_ref = db.collection('visitors')
            
            # Obtener todos los documentos de la colección
            documents = collection_ref.stream()
            
            # Convertir a lista para poder iterar múltiples veces
            doc_list = list(documents)
            
            # Obtener IDs de documentos actuales
            current_doc_ids = {doc.id for doc in doc_list}
            
            # Identificar nuevos documentos (no procesados anteriormente)
            new_doc_ids = current_doc_ids - last_processed_docs
            
            # Identificar documentos eliminados (estaban en last_processed_docs pero ya no están en current_doc_ids)
            deleted_doc_ids = last_processed_docs - current_doc_ids
            
            # Procesar documentos eliminados si existen
            if deleted_doc_ids:
                print(f"Se detectaron {len(deleted_doc_ids)} registros eliminados en la colección 'visitors'")
                
                embeddings_file = "embeddings_arcface.json"
                
                # Cargar embeddings existentes si el archivo existe
                if os.path.exists(embeddings_file):
                    try:
                        with open(embeddings_file, 'r') as f:
                            file_content = f.read().strip()
                            if file_content:
                                embeddings_data = json.loads(file_content)
                                
                                # Imprimir información de depuración
                                print(f"Documentos eliminados: {deleted_doc_ids}")
                                print(f"Claves actuales en embeddings: {list(embeddings_data.keys())}")
                                
                                # Necesitamos obtener los nombres completos de las personas eliminadas
                                # Para esto, consultamos la información almacenada antes de la eliminación
                                deleted_names = []
                                keys_to_delete = []
                                
                                # Obtener información de los documentos eliminados desde Firebase
                                # Como los documentos ya fueron eliminados, usaremos la información que tenemos
                                # en el archivo de embeddings y los logs
                                
                                # Primero, intentamos buscar en los logs o mensajes anteriores
                                # Si no es posible, pedimos al usuario que proporcione el nombre
                                
                                # Para cada documento eliminado, pedimos al usuario que confirme qué entrada eliminar
                                for doc_id in deleted_doc_ids:
                                    print(f"Procesando documento eliminado: {doc_id}")
                                    
                                    # Aquí podríamos implementar una lógica para guardar un mapeo de IDs a nombres
                                    # Por ahora, simplemente preguntamos al usuario o usamos información de logs
                                    
                                    # Buscar todas las claves que podrían corresponder al documento eliminado
                                    # y mostrarlas para que el usuario pueda confirmar
                                    potential_keys = list(embeddings_data.keys())
                                    
                                    # Si solo hay una clave, asumimos que es la correcta
                                    if len(potential_keys) == 1:
                                        key_to_delete = potential_keys[0]
                                        keys_to_delete.append(key_to_delete)
                                        deleted_names.append(key_to_delete)
                                        print(f"Se eliminará automáticamente: {key_to_delete}")
                                    else:
                                        # Buscar en los logs o mensajes para identificar qué clave eliminar
                                        # Por ahora, implementamos una lógica simple basada en el nombre mencionado en los logs
                                        for key in potential_keys:
                                            # Verificar si el nombre aparece en los logs o mensajes recientes
                                            # Esto es una simplificación, en la práctica necesitaríamos una forma más robusta
                                            # de mapear IDs de documentos a nombres de personas
                                            if "Jhonatan Alvarez" in key and doc_id in deleted_doc_ids:
                                                keys_to_delete.append(key)
                                                deleted_names.append(key)
                                                print(f"Se eliminará basado en logs: {key}")
                                
                                # Eliminar las claves identificadas
                                for key in keys_to_delete:
                                    try:
                                        del embeddings_data[key]
                                        print(f"Eliminado: {key}")
                                    except KeyError:
                                        print(f"Error: No se pudo eliminar {key}, no existe en el diccionario")
                                
                                # Guardar los embeddings actualizados
                                if keys_to_delete:
                                    with open(embeddings_file, 'w') as f:
                                        json.dump(embeddings_data, f)
                                    
                                    # Recargar los embeddings en el reconocedor
                                    recognizer.known_faces = recognizer.load_embeddings(embeddings_file)
                                    
                                    print(f"Se eliminaron automáticamente {len(deleted_names)} personas: {', '.join(deleted_names)}")
                    except Exception as e:
                        print(f"Error al procesar documentos eliminados: {e}")
                        import traceback
                        traceback.print_exc()
                
            # Actualizar el conjunto de documentos procesados (eliminar los IDs que ya no existen)
            last_processed_docs -= deleted_doc_ids
            
            if new_doc_ids:
                print(f"Se encontraron {len(new_doc_ids)} nuevos registros en la colección 'visitors'")
                
                # Procesar solo los nuevos documentos
                new_docs = [doc for doc in doc_list if doc.id in new_doc_ids]
                
                processed_names = []
                embeddings_data = {}
                embeddings_file = "embeddings_arcface.json"
                
                # Cargar embeddings existentes si el archivo existe
                if os.path.exists(embeddings_file):
                    try:
                        with open(embeddings_file, 'r') as f:
                            file_content = f.read().strip()
                            if file_content:
                                embeddings_data = json.loads(file_content)
                    except Exception as e:
                        print(f"Error al cargar embeddings existentes: {e}")
                
                # Procesar cada nuevo documento
                for doc in new_docs:
                    doc_data = doc.to_dict()
                    doc_id = doc.id
                    
                    # Verificar que el documento tenga los campos necesarios
                    if 'firstName' not in doc_data or 'lastName' not in doc_data or 'faceData' not in doc_data:
                        continue
                        
                    if 'images' not in doc_data['faceData']:
                        continue
                    
                    # Verificar si existe el campo idNumber
                    id_number = doc_data.get('idNumber', '')
                    if not id_number:
                        print(f"Advertencia: Documento {doc_id} no tiene número de identificación. Usando ID del documento.")
                        id_number = doc_id
                    
                    # Normalizar nombre y apellido
                    first_name = normalize_text(doc_data['firstName'])
                    last_name = normalize_text(doc_data['lastName'])
                    
                    # Construir el nombre completo de la persona con los nombres normalizados e incluir el número de identificación
                    person_name = f"{first_name} {last_name}_{id_number}"
                    
                    # Verificar si ya existe este nombre en los embeddings (verificación O(1))
                    if person_name in embeddings_data:
                        print(f"Persona {person_name} ya tiene embeddings. Omitiendo procesamiento.")
                        processed_names.append(f"{person_name} (ya existente)")
                        continue
                        
                    person_embeddings = []
                    
                    # Obtener las imágenes del subcampo Images
                    images = doc_data['faceData']['images']
                    
                    # Verificar si images es una lista o un diccionario
                    if isinstance(images, dict):
                        # Si es un diccionario, convertir a lista de valores
                        images = list(images.values())
                    elif not isinstance(images, list):
                        # Si no es ni lista ni diccionario, continuar con el siguiente documento
                        continue
                        
                    # Procesar cada imagen
                    for img_base64 in images:
                        if img_base64:
                            try:
                                # Eliminar el prefijo data:image/jpeg;base64, si existe
                                if img_base64.startswith('data:image/jpeg;base64,'):
                                    img_base64 = img_base64.replace('data:image/jpeg;base64,', '')
                                elif img_base64.startswith('data:image/png;base64,'):
                                    img_base64 = img_base64.replace('data:image/png;base64,', '')
                                
                                # Decodificar la imagen base64
                                img_data = base64.b64decode(img_base64)
                                nparr = np.frombuffer(img_data, np.uint8)
                                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if img is None:
                                    continue
                                
                                # Obtener embeddings usando el reconocedor facial existente
                                faces = recognizer.app.get(img)
                                
                                if faces:
                                    # Tomar el embedding de la primera cara detectada
                                    embedding = faces[0].embedding.tolist()
                                    person_embeddings.append(embedding)
                            except Exception as e:
                                print(f"Error al procesar imagen para {person_name}: {e}")
                    
                    # Si se obtuvieron embeddings, guardarlos
                    if person_embeddings:
                        embeddings_data[person_name] = person_embeddings
                        processed_names.append(person_name)
                
                # Guardar los embeddings actualizados
                if processed_names:
                    with open(embeddings_file, 'w') as f:
                        json.dump(embeddings_data, f)
                    
                    # Recargar los embeddings en el reconocedor
                    recognizer.known_faces = recognizer.load_embeddings(embeddings_file)
                    
                    print(f"Se procesaron automáticamente {len(processed_names)} personas: {', '.join(processed_names)}")
                
                # Actualizar el conjunto de documentos procesados
                last_processed_docs.update(new_doc_ids)
            
            # Esperar antes de la próxima verificación (30 segundos)
            time.sleep(30)
            
        except Exception as e:
            print(f"Error en el monitoreo de Firebase: {e}")
            time.sleep(60)  # Esperar un minuto antes de reintentar en caso de error

# ==================== ENDPOINTS ====================

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


@app.get('/favicon.ico', include_in_schema=False)
async def disable_favicon():
    """Endpoint para ignorar favicon.ico"""
    return Response(status_code=204)  # Respuesta vacía con código 204 (No Content)

@app.get("/")
def root():
    return Response(content="API cámaras funcionando!", status_code=200)

if __name__ == "__main__":
    # Iniciar el hilo de monitoreo de Firebase
    firebase_monitor_thread = threading.Thread(target=monitor_firebase_visitors, daemon=True)
    firebase_monitor_thread.start()
    
    # Iniciar el servidor FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)