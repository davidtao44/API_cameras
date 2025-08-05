from typing import Dict, List
from models.relay import RelayConfig, RelayType

# Configuración estática de relés
RELAY_CONFIGURATIONS: Dict[str, RelayConfig] = {
    "relay_001": RelayConfig(
        name="Relé Entrada Principal",
        description="Control de acceso para la entrada principal del edificio",
        relay_type=RelayType.HTTP,
        config={
            "ip_address": "172.16.2.47",
            "port": 80,
            "channel": 0,
            "auth": None,
            "timeout": 10,
            "use_rpc": True
        }
    ),
    "relay_002": RelayConfig(
        name="Relé Alarma",
        description="Control de alarma para la entrada principal del edificio",
        relay_type=RelayType.HTTP,
        config={
            "ip_address": "172.16.2.43",
            "port": 80,
            "channel": 0,
            "auth": None,
            "timeout": 10,
            "use_rpc": True
        }
    )
}

def get_relay_config(relay_id: str) -> RelayConfig:
    """Obtener configuración de un relé específico"""
    if relay_id not in RELAY_CONFIGURATIONS:
        raise ValueError(f"Relé {relay_id} no encontrado en la configuración")
    return RELAY_CONFIGURATIONS[relay_id]

def get_all_relay_configs() -> Dict[str, RelayConfig]:
    """Obtener todas las configuraciones de relés"""
    return RELAY_CONFIGURATIONS.copy()

def get_relay_list() -> List[Dict[str, str]]:
    """Obtener lista simplificada de relés con información básica"""
    return [
        {
            "relay_id": relay_id,
            "name": config.name,
            "description": config.description,
            "ip_address": config.config.get("ip_address", "N/A"),
            "relay_type": config.relay_type.value
        }
        for relay_id, config in RELAY_CONFIGURATIONS.items()
    ]

def relay_exists(relay_id: str) -> bool:
    """Verificar si un relé existe en la configuración"""
    return relay_id in RELAY_CONFIGURATIONS