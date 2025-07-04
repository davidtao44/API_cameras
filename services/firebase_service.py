import base64
import json
import os
import time
import numpy as np
from datetime import datetime
from config.firebase import db
from utils.normalizers import normalize_text
from services.camera_stream import stream_recognizer
import cv2

# Variable para almacenar los últimos documentos procesados
last_processed_docs = set()

def monitor_firebase_visitors(firebase_initialized: bool):
    """
    Función que monitorea constantemente la colección 'visitors' de Firebase
    para detectar nuevos registros y generar embeddings automáticamente.
    """
    global last_processed_docs
    
    print("Iniciando monitoreo de la colección 'visitors' en Firebase...")
    
    mapping_file = "firebase_mapping.json"
    firebase_mapping = load_mapping_file(mapping_file)
    
    while True:
        try:
            if not firebase_initialized:
                print("Firebase no está inicializado. Reintentando en 60 segundos...")
                time.sleep(60)
                continue
                
            collection_ref = db.collection('visitors')
            documents = collection_ref.stream()
            doc_list = list(documents)
            
            current_doc_ids = {doc.id for doc in doc_list}
            new_doc_ids = current_doc_ids - last_processed_docs
            deleted_doc_ids = last_processed_docs - current_doc_ids
            
            if deleted_doc_ids:
                process_deleted_documents(deleted_doc_ids, firebase_mapping, mapping_file)
            
            last_processed_docs -= deleted_doc_ids
            
            if new_doc_ids:
                process_new_documents(doc_list, new_doc_ids, firebase_mapping, mapping_file)
                last_processed_docs.update(new_doc_ids)
            
            time.sleep(30)
            
        except Exception as e:
            print(f"Error en el monitoreo de Firebase: {e}")
            time.sleep(60)

def load_mapping_file(mapping_file: str) -> dict:
    """Carga el archivo de mapeo de Firebase si existe"""
    firebase_mapping = {}
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                file_content = f.read().strip()
                if file_content:
                    firebase_mapping = json.loads(file_content)
        except Exception as e:
            print(f"Error al cargar el archivo de mapeo: {e}")
    return firebase_mapping

def process_deleted_documents(deleted_doc_ids: set, firebase_mapping: dict, mapping_file: str):
    """Procesa documentos eliminados de Firebase"""
    print(f"Se detectaron {len(deleted_doc_ids)} registros eliminados en la colección 'visitors'")
    embeddings_file = "embeddings_arcface.json"
    
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'r') as f:
                file_content = f.read().strip()
                if file_content:
                    embeddings_data = json.loads(file_content)
                    
                    deleted_names = []
                    keys_to_delete = []
                    
                    for doc_id in deleted_doc_ids:
                        if doc_id in firebase_mapping:
                            key_to_delete = firebase_mapping[doc_id]
                            if key_to_delete in embeddings_data:
                                keys_to_delete.append(key_to_delete)
                                deleted_names.append(key_to_delete)
                    
                    for key in keys_to_delete:
                        try:
                            del embeddings_data[key]
                            for doc_id, mapped_key in list(firebase_mapping.items()):
                                if mapped_key == key:
                                    del firebase_mapping[doc_id]
                        except KeyError:
                            continue
                    
                    if keys_to_delete:
                        with open(embeddings_file, 'w') as f:
                            json.dump(embeddings_data, f)
                        
                        with open(mapping_file, 'w') as f:
                            json.dump(firebase_mapping, f)
                        
                        # En la línea 108 - función process_deleted_documents
                        stream_recognizer.known_faces = stream_recognizer.load_embeddings(embeddings_file)
                        
                        print(f"Se eliminaron automáticamente {len(deleted_names)} personas: {', '.join(deleted_names)}")
        except Exception as e:
            print(f"Error al procesar documentos eliminados: {e}")

def process_new_documents(doc_list: list, new_doc_ids: set, firebase_mapping: dict, mapping_file: str):
    """Procesa nuevos documentos de Firebase"""
    print(f"Se encontraron {len(new_doc_ids)} nuevos registros en la colección 'visitors'")
    new_docs = [doc for doc in doc_list if doc.id in new_doc_ids]
    
    processed_names = []
    embeddings_data = {}
    embeddings_file = "embeddings_arcface.json"
    
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'r') as f:
                file_content = f.read().strip()
                if file_content:
                    embeddings_data = json.loads(file_content)
        except Exception as e:
            print(f"Error al cargar embeddings existentes: {e}")
    
    for doc in new_docs:
        doc_data = doc.to_dict()
        doc_id = doc.id
        
        if 'firstName' not in doc_data or 'lastName' not in doc_data or 'faceData' not in doc_data:
            continue
            
        if 'images' not in doc_data['faceData']:
            continue
        
        id_number = doc_data.get('idNumber', doc_id)
        first_name = normalize_text(doc_data['firstName'])
        last_name = normalize_text(doc_data['lastName'])
        person_name = f"{first_name} {last_name}_{id_number}"
        
        firebase_mapping[doc_id] = person_name
        
        if person_name in embeddings_data:
            processed_names.append(f"{person_name} (ya existente)")
            continue
            
        person_embeddings = process_person_images(doc_data, person_name)
        
        if person_embeddings:
            embeddings_data[person_name] = person_embeddings
            processed_names.append(person_name)
    
    if processed_names:
        save_embeddings_and_mapping(embeddings_file, embeddings_data, mapping_file, firebase_mapping)
        print(f"Se procesaron automáticamente {len(processed_names)} personas: {', '.join(processed_names)}")

def process_person_images(doc_data: dict, person_name: str) -> list:
    """Procesa las imágenes de una persona para generar embeddings"""
    person_embeddings = []
    images = doc_data['faceData']['images']
    
    if isinstance(images, dict):
        images = list(images.values())
    elif not isinstance(images, list):
        return []
        
    for img_base64 in images:
        if img_base64:
            try:
                img_base64 = clean_base64_prefix(img_base64)
                img_data = base64.b64decode(img_base64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # En la línea 184 - función process_person_images
                faces = stream_recognizer.app.get(img)
                if faces:
                    embedding = faces[0].embedding.tolist()
                    person_embeddings.append(embedding)
            except Exception as e:
                print(f"Error al procesar imagen para {person_name}: {e}")
    
    return person_embeddings

def clean_base64_prefix(img_base64: str) -> str:
    """Limpia el prefijo de datos de una cadena base64"""
    if img_base64.startswith('data:image/jpeg;base64,'):
        return img_base64.replace('data:image/jpeg;base64,', '')
    elif img_base64.startswith('data:image/png;base64,'):
        return img_base64.replace('data:image/png;base64,', '')
    return img_base64

def save_embeddings_and_mapping(embeddings_file: str, embeddings_data: dict, 
                               mapping_file: str, firebase_mapping: dict):
    """Guarda los embeddings y el mapeo actualizados"""
    with open(embeddings_file, 'w') as f:
        json.dump(embeddings_data, f)
    
    with open(mapping_file, 'w') as f:
        json.dump(firebase_mapping, f)
    
    stream_recognizer.known_faces = stream_recognizer.load_embeddings(embeddings_file)