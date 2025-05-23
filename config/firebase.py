import firebase_admin
from firebase_admin import credentials, firestore
import os

# Variable global para controlar la inicialización
_firebase_initialized = False

def initialize_firebase():
    """Inicializa la conexión con Firebase solo una vez"""
    global _firebase_initialized
    
    if _firebase_initialized:
        return True
        
    try:
        cred_path = r"/home/tecon/Documentos/Camara/server_video/keys/serviceAccountKey.json"
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
        return True
    except Exception as e:
        print(f"Error al inicializar Firebase: {e}")
        return False

# Inicializar Firebase una sola vez al importar el módulo
initialize_firebase()

# Exportar la instancia de la base de datos
db = firestore.client() if _firebase_initialized else None