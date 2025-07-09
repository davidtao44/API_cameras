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
        # Fixed path for Docker container
        cred_path = "/app/keys/serviceAccountKey.json"
        # Fallback for local development
        if not os.path.exists(cred_path):
            cred_path = os.path.join(os.path.dirname(__file__), "..", "..", "keys", "serviceAccountKey.json")
        
        print(f"🔑 Intentando cargar credenciales desde: {cred_path}")
        
        if not os.path.exists(cred_path):
            raise FileNotFoundError(f"No se encontró el archivo de credenciales en: {cred_path}")
            
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
        print("✅ Firebase inicializado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error al inicializar Firebase: {e}")
        return False

# Inicializar Firebase una sola vez al importar el módulo
initialize_firebase()

# Exportar la instancia de la base de datos
db = firestore.client() if _firebase_initialized else None