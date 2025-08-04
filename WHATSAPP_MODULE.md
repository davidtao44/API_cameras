# 📱 Módulo de Notificaciones WhatsApp

Este módulo permite enviar notificaciones automáticas por WhatsApp cuando se detectan eventos de intrusión en las cámaras de seguridad.

## 🚀 Características

- **Notificaciones automáticas**: Envío automático de alertas cuando se detectan intrusiones
- **Videos adjuntos**: Creación y envío de videos desde el buffer de frames procesados
- **Configuración por cámara**: Control individual de notificaciones para cada cámara
- **Cooldown inteligente**: Prevención de spam con intervalos configurables
- **Integración EvoAPI**: Utiliza EvoAPI para el envío de mensajes WhatsApp

## ⚙️ Configuración

### Variables de Entorno

Configura las siguientes variables de entorno para EvoAPI:

```bash
export EVOAPI_URL="https://api.evoapi.com"
export EVOAPI_KEY="tu_clave_de_evoapi"
```

### Configuración por Cámara

En `config/settings.py`, cada cámara tiene las siguientes opciones:

```python
"cam1": {
    "whatsapp_notifications_enabled": True,  # Habilitar/deshabilitar notificaciones
    "notification_phones": ['573107834112', '573222262665'],  # Números de teléfono
    "video_duration_seconds": 10,  # Duración del video en segundos
    # ... otras configuraciones
}
```

## 📡 Endpoints Disponibles

### Estado del Servicio

#### `GET /whatsapp/status`
Obtiene el estado general del servicio WhatsApp.

**Respuesta:**
```json
{
    "success": true,
    "status": {
        "service_active": true,
        "evoapi_configured": true,
        "notification_cooldown": 300,
        "cameras_with_buffers": ["cam1", "cam2", "cam3"],
        "buffer_status": {
            "cam1": {
                "frame_count": 45,
                "max_duration": 10
            }
        }
    }
}
```

#### `GET /whatsapp/status/{camera_id}`
Obtiene el estado de WhatsApp para una cámara específica.

### Configuración

#### `POST /whatsapp/config`
Actualiza la configuración del servicio WhatsApp.

**Body:**
```json
{
    "evoapi_url": "https://api.evoapi.com",
    "api_key": "nueva_clave",
    "notification_cooldown": 300
}
```

### Notificaciones

#### `POST /whatsapp/send-manual`
Envía una notificación manual con video.

**Body:**
```json
{
    "camera_id": "cam1",
    "message": "Mensaje personalizado",
    "phone_numbers": ["573107834112"],
    "include_video": true,
    "video_duration": 10
}
```

#### `POST /whatsapp/test-notification`
Envía una notificación de prueba.

**Body:**
```json
{
    "camera_id": "cam1",
    "event_type": "face_mismatch",
    "person_count": 2,
    "face_count": 1
}
```

### Utilidades

#### `GET /whatsapp/cameras-config`
Obtiene la configuración de WhatsApp para todas las cámaras.

#### `POST /whatsapp/create-video/{camera_id}`
Crea un video desde el buffer de frames (para pruebas).

#### `GET /whatsapp/buffer-status`
Obtiene el estado de todos los buffers de video.

## 🔄 Funcionamiento Automático

### Detección de Intrusiones

El módulo se integra automáticamente con el sistema de detección de intrusiones y envía notificaciones cuando:

1. **Face Mismatch**: Se detectan más personas que rostros
2. **Unknown Person**: Se detecta una persona desconocida por tiempo prolongado

### Proceso de Notificación

1. **Detección**: El sistema detecta una condición de intrusión
2. **Timer**: Se inicia un countdown configurable
3. **Verificación**: Al expirar el timer, se verifica si la condición persiste
4. **Video**: Se crea un video desde el buffer de frames procesados
5. **Envío**: Se envía la notificación WhatsApp con el video adjunto
6. **Cooldown**: Se activa un período de espera antes de la próxima notificación

### Formato del Mensaje

```
🚨 *ALERTA DE SEGURIDAD* 🚨

📹 Cámara: CAM1
⚠️ Tipo: Discrepancia de Rostros
📊 Detalles: Se detectaron 2 personas pero solo 1 rostros
🕐 Hora: 2024-01-15 14:30:25

📹 Video adjunto de los últimos 10 segundos
```

## 🛠️ Arquitectura Técnica

### Componentes Principales

1. **VideoBuffer**: Almacena frames procesados en memoria para cada cámara
2. **WhatsAppService**: Servicio principal que maneja la lógica de notificaciones
3. **Frame Collector**: Hilo en segundo plano que recolecta frames procesados
4. **Video Creator**: Crea videos MP4 desde los frames almacenados
5. **EvoAPI Integration**: Interfaz con la API de WhatsApp

### Buffer de Frames

- **Duración**: Configurable por cámara (default: 10 segundos)
- **FPS**: 10 frames por segundo para los videos
- **Limpieza**: Automática de frames antiguos
- **Thread-Safe**: Uso de locks para acceso concurrente

### Cooldown System

- **Por cámara**: Cada cámara tiene su propio cooldown
- **Configurable**: Default 5 minutos entre notificaciones
- **Prevención de spam**: Evita notificaciones excesivas

## 🔧 Troubleshooting

### Problemas Comunes

#### No se envían notificaciones
1. Verificar que `whatsapp_notifications_enabled` esté en `true`
2. Comprobar que hay números en `notification_phones`
3. Verificar la configuración de EvoAPI
4. Revisar logs del servidor

#### Videos no se crean
1. Verificar que hay frames en el buffer
2. Comprobar permisos de escritura en `/tmp`
3. Verificar instalación de OpenCV

#### Errores de EvoAPI
1. Verificar la URL de EvoAPI
2. Comprobar la validez de la API key
3. Verificar conectividad de red

### Logs Importantes

```bash
[INFO] WhatsApp Service initialized with EvoAPI
[INFO] WhatsApp notification sent for cam1
[WARNING] WhatsApp notifications disabled for cam2
[ERROR] WhatsApp notification error for cam1: API key invalid
```

## 📊 Monitoreo

### Métricas Disponibles

- Estado del servicio
- Número de frames en buffer por cámara
- Última notificación enviada por cámara
- Estado de cooldown
- Configuración de EvoAPI

### Health Check

Usa `GET /whatsapp/status` para verificar el estado del servicio:

```bash
curl http://localhost:8000/whatsapp/status
```

## 🔐 Seguridad

- **API Keys**: Nunca hardcodear claves en el código
- **Variables de entorno**: Usar para configuración sensible
- **Validación**: Todos los endpoints validan entrada
- **Rate limiting**: Cooldown previene abuso
- **Cleanup**: Videos temporales se eliminan automáticamente

## 📝 Ejemplo de Uso

### Configuración Inicial

1. Configurar variables de entorno:
```bash
export EVOAPI_URL="https://api.evoapi.com"
export EVOAPI_KEY="tu_clave_aqui"
```

2. Habilitar notificaciones en `settings.py`:
```python
"cam1": {
    "whatsapp_notifications_enabled": True,
    "notification_phones": ['573107834112'],
    "video_duration_seconds": 15
}
```

3. Reiniciar el servidor

### Prueba Manual

```bash
# Enviar notificación de prueba
curl -X POST http://localhost:8000/whatsapp/test-notification \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "cam1",
    "event_type": "face_mismatch",
    "person_count": 2,
    "face_count": 1
  }'
```

## 🚀 Próximas Mejoras

- [ ] Soporte para mensajes de texto sin video
- [ ] Plantillas de mensajes personalizables
- [ ] Integración con otros proveedores de WhatsApp
- [ ] Dashboard web para gestión de notificaciones
- [ ] Estadísticas de entrega de mensajes
- [ ] Soporte para grupos de WhatsApp