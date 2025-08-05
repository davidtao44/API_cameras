"""Configuración de optimización para resolver problemas de rendimiento

Este archivo contiene configuraciones optimizadas para:
1. Reducir el bloqueo del stream por operaciones síncronas
2. Ajustar los tiempos de persistencia de tracking de personas
"""

# ==================== CONFIGURACIÓN DE TRACKING ====================

# Configuración de tracking optimizada con persistencia corta
TRACKING_CONFIG = {
    # Persistencia corta - 3 segundos
    'track_max_age': 3,
    
    # Persistencia corta para reconocimiento facial - 2 segundos
    'face_persistence_time': 2,
    
    # Umbral IoU para asociación de detecciones (mantenido)
    'iou_threshold': 0.3,
    
    # Distancia máxima para asociar cuerpo con cara (reducido para mayor precisión)
    'max_distance_threshold': 80,
    
    # Intervalo de limpieza de tracks expirados (más frecuente)
    'cleanup_interval': 2.0,
    
    # Modo tracking normal con persistencia corta
    'frame_by_frame_mode': False
}

# ==================== CONFIGURACIÓN DE PROCESAMIENTO ====================

# Configuración para el procesamiento de frames
PROCESSING_CONFIG = {
    # Tamaño máximo de cola por cámara (reducido para menor latencia)
    'max_queue_size': 2,
    
    # Intervalo entre análisis de frames en el stream (aumentado para reducir carga)
    'analysis_interval': 0.8,  # 800ms entre análisis
    
    # Timeout para operaciones de procesamiento
    'processing_timeout': 1.0,
    
    # Número máximo de hilos de procesamiento por cámara
    'max_processing_threads': 1
}

# ==================== CONFIGURACIÓN DE ENDPOINTS ====================

# Configuración para endpoints que pueden bloquear el stream
ENDPOINT_CONFIG = {
    # Timeout para operaciones de video de WhatsApp
    'whatsapp_video_timeout': 30.0,
    
    # Ejecutar operaciones pesadas en hilos separados
    'async_video_creation': True,
    
    # Timeout para operaciones de acceso control
    'access_control_timeout': 2.0,
    
    # Cache de resultados para endpoints de estadísticas
    'stats_cache_duration': 1.0
}

# ==================== CONFIGURACIÓN DE STREAMING ====================

# Configuración para optimizar el streaming
STREAMING_CONFIG = {
    # Calidad de compresión JPEG para el stream (reducida para mayor velocidad)
    'jpeg_quality': 75,
    
    # Tamaño de chunk para streaming
    'chunk_size': 1024,
    
    # Timeout para conexiones de cámara
    'camera_timeout': 5.0,
    
    # Intervalo mínimo entre frames del stream
    'min_frame_interval': 0.033  # ~30 FPS máximo
}

# ==================== CONFIGURACIÓN DE CACHE ====================

# Configuración de cache para resultados de procesamiento
CACHE_CONFIG = {
    # Duración del cache de resultados de procesamiento
    'processing_result_cache_duration': 2.0,
    
    # Duración del cache de estadísticas
    'stats_cache_duration': 1.0,
    
    # Duración del cache de control de acceso
    'access_control_cache_duration': 2.0
}

# ==================== FUNCIONES DE APLICACIÓN ====================

def apply_tracking_optimizations():
    """Aplica las optimizaciones de tracking al servicio correspondiente"""
    try:
        from surveillance.person_tracking_service import person_tracking_service
        
        # Aplicar configuraciones optimizadas
        person_tracking_service.track_max_age = TRACKING_CONFIG['track_max_age']
        person_tracking_service.face_persistence_time = TRACKING_CONFIG['face_persistence_time']
        person_tracking_service.iou_threshold = TRACKING_CONFIG['iou_threshold']
        person_tracking_service.max_distance_threshold = TRACKING_CONFIG['max_distance_threshold']
        person_tracking_service.set_frame_by_frame_mode(TRACKING_CONFIG['frame_by_frame_mode'])
        
        print("[INFO] Optimizaciones de tracking aplicadas:")
        print(f"  - track_max_age: {TRACKING_CONFIG['track_max_age']}s")
        print(f"  - face_persistence_time: {TRACKING_CONFIG['face_persistence_time']}s")
        print(f"  - iou_threshold: {TRACKING_CONFIG['iou_threshold']}")
        print(f"  - max_distance_threshold: {TRACKING_CONFIG['max_distance_threshold']}px")
        print(f"  - frame_by_frame_mode: {TRACKING_CONFIG['frame_by_frame_mode']}")
        
    except ImportError as e:
        print(f"[WARNING] No se pudo aplicar optimizaciones de tracking: {e}")

def apply_processing_optimizations():
    """Aplica las optimizaciones de procesamiento al servicio correspondiente"""
    try:
        from surveillance.enhanced_frame_processing_service import enhanced_frame_processing_service
        
        # Aplicar configuraciones optimizadas
        enhanced_frame_processing_service.max_queue_size = PROCESSING_CONFIG['max_queue_size']
        
        print("[INFO] Optimizaciones de procesamiento aplicadas:")
        print(f"  - max_queue_size: {PROCESSING_CONFIG['max_queue_size']}")
        print(f"  - analysis_interval: {PROCESSING_CONFIG['analysis_interval']}s")
        
    except ImportError as e:
        print(f"[WARNING] No se pudo aplicar optimizaciones de procesamiento: {e}")

def apply_all_optimizations():
    """Aplica todas las optimizaciones disponibles"""
    print("[INFO] Aplicando optimizaciones de rendimiento...")
    apply_tracking_optimizations()
    apply_processing_optimizations()
    print("[INFO] Optimizaciones aplicadas exitosamente")