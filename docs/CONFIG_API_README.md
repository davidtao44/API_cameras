# API de Configuración de Cámaras

Este documento describe los endpoints disponibles para gestionar la configuración de las cámaras del sistema de vigilancia.

## Endpoints Disponibles

### 1. Obtener configuración de todas las cámaras

**GET** `/config/cameras`

Retorna la configuración de control para todas las cámaras registradas.

**Respuesta:**
```json
{
  "cameras": [
    {
      "camera_id": "cam1",
      "access_control_enabled": true,
      "alert_control_enabled": true,
      "whatsapp_notifications_enabled": true
    },
    {
      "camera_id": "cam2",
      "access_control_enabled": false,
      "alert_control_enabled": false,
      "whatsapp_notifications_enabled": false
    }
  ]
}
```

### 2. Obtener configuración de una cámara específica

**GET** `/config/cameras/{camera_id}`

Retorna la configuración de control para una cámara específica.

**Parámetros:**
- `camera_id` (string): ID de la cámara (ej: "cam1", "cam2", "cam3")

**Respuesta:**
```json
{
  "camera_id": "cam1",
  "access_control_enabled": true,
  "alert_control_enabled": true,
  "whatsapp_notifications_enabled": true
}
```

### 3. Actualizar configuración de una cámara

**PUT** `/config/cameras/{camera_id}`

Actualiza la configuración de control para una cámara específica.

**Parámetros:**
- `camera_id` (string): ID de la cámara

**Body (JSON):**
```json
{
  "access_control_enabled": true,
  "alert_control_enabled": false,
  "whatsapp_notifications_enabled": true
}
```

**Nota:** Todos los campos son opcionales. Solo se actualizarán los campos proporcionados.

**Respuesta:**
```json
{
  "message": "Configuración actualizada para cam1",
  "updated_fields": ["access_control_enabled", "whatsapp_notifications_enabled"],
  "current_config": {
    "camera_id": "cam1",
    "access_control_enabled": true,
    "alert_control_enabled": true,
    "whatsapp_notifications_enabled": true
  }
}
```

## Descripción de los Campos de Configuración

### access_control_enabled
- **Tipo:** Boolean
- **Descripción:** Control de acceso y envío de mensajes por HTTP
- **Efecto:** Cuando está habilitado, el sistema procesará eventos de control de acceso y enviará notificaciones HTTP correspondientes

### alert_control_enabled
- **Tipo:** Boolean
- **Descripción:** Control de alarma y bloqueo de peticiones al relé
- **Efecto:** Cuando está habilitado, el sistema procesará alertas y controlará el acceso al relé basado en las detecciones

### whatsapp_notifications_enabled
- **Tipo:** Boolean
- **Descripción:** Envío de alertas por WhatsApp
- **Efecto:** Cuando está habilitado, el sistema enviará notificaciones de WhatsApp a los números configurados para la cámara

## Ejemplos de Uso

### Ejemplo 1: Habilitar solo notificaciones de WhatsApp para cam1
```bash
curl -X PUT "http://localhost:8000/config/cameras/cam1" \
  -H "Content-Type: application/json" \
  -d '{
    "whatsapp_notifications_enabled": true
  }'
```

### Ejemplo 2: Deshabilitar todas las funciones para cam2
```bash
curl -X PUT "http://localhost:8000/config/cameras/cam2" \
  -H "Content-Type: application/json" \
  -d '{
    "access_control_enabled": false,
    "alert_control_enabled": false,
    "whatsapp_notifications_enabled": false
  }'
```

### Ejemplo 3: Obtener configuración actual de todas las cámaras
```bash
curl -X GET "http://localhost:8000/config/cameras"
```

## Códigos de Error

- **404 Not Found:** La cámara especificada no existe
- **400 Bad Request:** No se proporcionaron campos para actualizar
- **500 Internal Server Error:** Error al guardar la configuración en el archivo

## Persistencia

Los cambios realizados a través de esta API se guardan automáticamente en el archivo `config/settings.py`, por lo que persisten entre reinicios del servidor.