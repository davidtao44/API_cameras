from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class RelayStatus(str, Enum):
    """Estados posibles del relé"""
    ON = "on"
    OFF = "off"
    UNKNOWN = "unknown"

class RelayType(str, Enum):
    """Tipos de relé soportados"""
    HTTP = "HTTP"
    MODBUS = "MODBUS"
    GPIO = "GPIO"
    SERIAL = "SERIAL"

class RelayConfig(BaseModel):
    """Configuración base para relés"""
    name: str = Field(..., description="Nombre del relé")
    description: Optional[str] = Field(None, description="Descripción del relé")
    relay_type: RelayType = Field(..., description="Tipo de relé")
    config: Dict[str, Any] = Field(..., description="Configuración específica del tipo de relé")

class HTTPRelayConfig(BaseModel):
    """Configuración específica para relés HTTP (Shelly)"""
    ip_address: str = Field(..., description="Dirección IP del dispositivo")
    port: int = Field(80, description="Puerto HTTP")
    channel: int = Field(0, description="Canal del relé (0 para Shelly Plus 1 PM)")
    auth: Optional[Dict[str, str]] = Field(None, description="Credenciales de autenticación")
    timeout: int = Field(10, description="Timeout en segundos")
    use_rpc: bool = Field(True, description="Usar protocolo RPC en lugar de HTTP directo")

class GPIORelayConfig(BaseModel):
    """Configuración para relés GPIO"""
    pin: int = Field(..., description="Pin GPIO")
    active_high: bool = Field(True, description="Lógica activa alta")

class ModbusRelayConfig(BaseModel):
    """Configuración para relés Modbus"""
    host: str = Field(..., description="Host Modbus")
    port: int = Field(502, description="Puerto Modbus")
    unit_id: int = Field(1, description="ID de unidad Modbus")
    register: int = Field(..., description="Registro de control")

class SerialRelayConfig(BaseModel):
    """Configuración para relés serie"""
    port: str = Field(..., description="Puerto serie")
    baudrate: int = Field(9600, description="Velocidad de baudios")
    command_on: str = Field(..., description="Comando para encender")
    command_off: str = Field(..., description="Comando para apagar")

class RelayState(BaseModel):
    """Estado actual del relé"""
    relay_id: str = Field(..., description="ID del relé")
    is_on: bool = Field(..., description="Estado del relé (encendido/apagado)")
    power_consumption: Optional[float] = Field(None, description="Consumo de potencia en watts")
    voltage: Optional[float] = Field(None, description="Voltaje en volts")
    current: Optional[float] = Field(None, description="Corriente en amperios")
    energy_total: Optional[float] = Field(None, description="Energía total consumida en Wh")
    temperature: Optional[float] = Field(None, description="Temperatura del dispositivo en °C")
    has_timer: bool = Field(False, description="Si tiene temporizador activo")
    timer_remaining: int = Field(0, description="Tiempo restante del temporizador en segundos")
    source: str = Field(..., description="Fuente de la información (http, rpc, gpio, etc.)")
    last_updated: Optional[str] = Field(None, description="Última actualización")

class RelayCommand(BaseModel):
    """Comando para controlar el relé"""
    action: RelayStatus = Field(..., description="Acción a realizar")
    timer_seconds: Optional[int] = Field(None, description="Temporizador en segundos")

class RelayResponse(BaseModel):
    """Respuesta estándar para operaciones de relé"""
    success: bool = Field(..., description="Si la operación fue exitosa")
    message: str = Field(..., description="Mensaje descriptivo")
    relay_id: str = Field(..., description="ID del relé")
    state: Optional[RelayState] = Field(None, description="Estado actual del relé")
    error_code: Optional[str] = Field(None, description="Código de error si aplica")

class RelayListResponse(BaseModel):
    """Respuesta para listar relés"""
    success: bool = Field(..., description="Si la operación fue exitosa")
    message: str = Field(..., description="Mensaje descriptivo")
    relays: List[Dict[str, Any]] = Field(..., description="Lista de relés configurados")
    total_count: int = Field(..., description="Número total de relés")

class RelayConfigResponse(BaseModel):
    """Respuesta para obtener configuración de relé"""
    success: bool = Field(..., description="Si la operación fue exitosa")
    message: str = Field(..., description="Mensaje descriptivo")
    relay_id: str = Field(..., description="ID del relé")
    config: Optional[RelayConfig] = Field(None, description="Configuración del relé")

class DeviceInfo(BaseModel):
    """Información del dispositivo Shelly"""
    id: str = Field(..., description="ID del dispositivo")
    mac: str = Field(..., description="Dirección MAC")
    model: str = Field(..., description="Modelo del dispositivo")
    gen: int = Field(..., description="Generación del dispositivo")
    fw_id: str = Field(..., description="ID del firmware")
    ver: str = Field(..., description="Versión del firmware")
    app: str = Field(..., description="Aplicación")
    name: Optional[str] = Field(None, description="Nombre del dispositivo")
    profile: Optional[str] = Field(None, description="Perfil activo")

class DeviceInfoResponse(BaseModel):
    """Respuesta para información del dispositivo"""
    success: bool = Field(..., description="Si la operación fue exitosa")
    message: str = Field(..., description="Mensaje descriptivo")
    relay_id: str = Field(..., description="ID del relé")
    device_info: Optional[DeviceInfo] = Field(None, description="Información del dispositivo")