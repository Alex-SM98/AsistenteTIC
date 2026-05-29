"""
============================================================================
CONFIGURACIÓN CENTRALIZADA - ASISTENTE TIC
============================================================================
Proyecto: TFG - Premio Don Bosco
Autor: Alex Simbaña
============================================================================
"""

import torch
import platform

# Configuración de modelos
MODELS = {
    'lse': {
        'path': 'models/best.pt',
        'name': 'Intérprete LSE',
        'description': 'Reconocimiento de Lengua de Signos Española',
        'classes': 30,
        'class_names': [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'GRACIAS', 'H', 'HOLA', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'POR FAVOR', 'Q', 'R', 'S', 'T', 'TE AMO', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ],
        'conf': 0.45,
        'iou': 0.3,
        'buffer_size': 7,
        'voice_interval': 1.5,
        'max_det': 5,
    },
    'visual': {
        'path': 'models/pedestrian_best.pt',
        'name': 'Asistente Visual',
        'description': 'Detección de semáforos peatonales',
        'classes': 4,
        'class_names': ['green', 'pedestrian Traffic Light', 'red', 'signal-light'],
        'conf': 0.4,
        'iou': 0.7,
        'buffer_size': 1,
        'voice_interval': 3.0,
        'max_det': 10,
    }
}

# Configuración de cámara
CAMERA = {
    'id': 0,
    'width': 640,
    'imgsz':320,
    'height': 480,
    'fps_target': 30,
}

# Configuración de voz
VOICE = {
    'enabled': True,
    'rate': 150,
    'volume': 1.0,
    'gender': 'male',
    'use_native_macos': platform.system() == 'Darwin',
    'jarvis_mode': True,
    'voice_id': 'europe/es',
}

# Configuración de dispositivo
DEVICE = {
    'type': 'mps' if torch.backends.mps.is_available() else 'cpu',
}

# Colores UI - Paleta Moderna y Elegante
COLORS = {
    'bg_dark': (18, 18, 18),           # Negro profundo
    'bg_panel': (30, 30, 30),          # Gris oscuro
    'bg_card': (42, 42, 42),           # Gris card
    'primary': (255, 165, 0),          # Naranja vibrante
    'success': (46, 204, 113),         # Verde esmeralda
    'warning': (241, 196, 15),         # Amarillo dorado
    'danger': (231, 76, 60),           # Rojo coral
    'info': (52, 152, 219),            # Azul cielo
    'text': (255, 255, 255),           # Blanco puro
    'text_dim': (149, 165, 166),       # Gris claro
    'accent': (155, 89, 182),          # Morado
    'border': (52, 73, 94),            # Azul grisáceo
}

# Mensajes del sistema
MESSAGES = {
    'startup': 'Bienvenido al Asistente TIC. Sistema KEYS iniciado. Por favor seleccione un modelo',
    'shutdown': 'Sistema KEYS finalizado. Hasta pronto.',
    'lse_startup': 'Módulo de interpretación de Lengua de Signos activado.',
    'visual_startup': 'Módulo de asistencia visual activado.',
    'red': 'Semáforo peatonal en rojo. Detente.',
    'green': 'Semáforo en verde. Puedes cruzar.',
    'voice_on': 'Síntesis de voz activada.',
    'voice_off': 'Síntesis de voz desactivada.',
}
