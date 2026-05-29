"""
============================================================================
ASISTENTE VISUAL CON VOZ
============================================================================
Sistema de detección de semáforos con Text-to-Speech simplificado.

Características:
- Detección en tiempo real
- Voz masculina/femenina
- Interfaz FPS y título

Controles:
- Q: Salir
- V: Voz ON/OFF
- M: Voz masculina
- F: Voz femenina

Autor: Alex Simbaña
Proyecto: TFG - Asistente Visual
============================================================================
"""

from ultralytics import YOLO
import torch
import cv2
import time
from datetime import datetime
from collections import defaultdict
from utils import CameraManager, VoiceEngine
from config import MESSAGES


class VisualAssistant:
    """Asistente Visual para detección de semáforos."""
    
    def __init__(self, model_config, camera_config, voice_config):
        self.model_config = model_config
        self.camera_config = camera_config
        self.voice_config = voice_config
        
        # Estado
        self.model = None
        self.voice = None
        self.camera = None
        self.stats = {'detections': 0}
        
        self.fps_start = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0

        self.frame_skip = 0
        self.last_annotated_frame = None
        
        self.window_name = "Asistente Visual - AsistenteTIC"
        
        self._init_model()
        self._init_voice()
    
    def _init_model(self):
        """Inicializa el modelo YOLO."""
        print(f"📦 Cargando modelo Asistente Visual...")
        try:
            self.model = YOLO(self.model_config['path'])
            print(f"   ✅ Modelo cargado: {self.model_config['classes']} clases")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            raise
    
    def _init_voice(self):
        """Inicializa el motor de voz."""
        self.voice = VoiceEngine(self.voice_config)

    def process_frame(self, frame):
        """Procesa un frame con YOLO con frame skip para optimizar FPS."""

        # Saltar frames para mejorar FPS en Raspberry Pi
        self.frame_skip += 1
        if self.frame_skip % 2 != 0:
            return self.last_annotated_frame if self.last_annotated_frame is not None else frame

        results = self.model(
            frame,
            verbose=False,
            conf=self.model_config['conf'],
            iou=self.model_config['iou'],
            imgsz=self.model_config.get('imgsz', 320),
        )

        annotated_frame = results[0].plot()

        # Procesar detecciones
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_name = self.model.names[int(box.cls[0])]
                if float(box.conf[0]) > self.model_config['conf']:
                    self.stats['detections'] += 1

                    if class_name in ['red', 'green']:
                        self.voice.speak(
                            MESSAGES[class_name],
                            min_interval=self.model_config['voice_interval']
                        )

        self.last_annotated_frame = annotated_frame
        return annotated_frame
    
    def draw_minimal_ui(self, frame):
        """Dibuja interfaz minimalista sobre el frame."""
        h, w = frame.shape[:2]
        
        # Fondo semi-transparente arriba
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Título del proyecto
        cv2.putText(frame, "TFG - ASISTENTE VISUAL", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Estado: Cámara ON + FPS
        status_text = f"Camara: ON  |  FPS: {self.current_fps:.1f}"
        cv2.putText(frame, status_text, 
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Voz (derecha)
        voice_color = (0, 255, 0) if self.voice.enabled else (0, 0, 255)
        voice_text = f"Voz: {'ON' if self.voice.enabled else 'OFF'} ({self.voice.gender[0].upper()})"
        text_size = cv2.getTextSize(voice_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(frame, voice_text, 
                    (w - text_size[0] - 10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)
        
        # Controles abajo (pequeños)
        controls = "Q: Salir  |  V: Voz  |  M: Masc  |  F: Fem"
        cv2.putText(frame, controls, 
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Ejecuta el asistente visual."""
        print("\n" + "=" * 70)
        print("👁️ ASISTENTE VISUAL CON VOZ")
        print("=" * 70)
        print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n💡 Controles:")
        print(f"   Q: Salir  |  V: Voz ON/OFF  |  M: Masc  |  F: Fem")
        print(f"\n🚀 Iniciando...\n")
        
        try:
            self.camera = CameraManager(
                self.camera_config['id'],
                self.camera_config['width'],
                self.camera_config['height']
            )
            
            while True:
                frame = self.camera.read_frame()
                if frame is None:
                    break
                
                self.fps_counter += 1
                
                # Calcular FPS cada segundo
                if time.time() - self.fps_start >= 1.0:
                    self.current_fps = self.fps_counter / (time.time() - self.fps_start)
                    self.fps_start = time.time()
                    self.fps_counter = 0
                
                # Detección
                results = self.model(
                    frame,
                    verbose=False,
                    conf=self.model_config['conf'],
                    iou=self.model_config['iou'],
                    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
                )
                
                # Frame anotado
                annotated_frame = results[0].plot()
                
                # Procesar detecciones
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        class_name = self.model.names[int(box.cls[0])]
                        if float(box.conf[0]) > self.model_config['conf']:
                            self.stats['detections'] += 1
                            
                            # Anunciar solo rojo y verde
                            if class_name in ['red', 'green']:
                                self.voice.speak(
                                    MESSAGES[class_name], 
                                    min_interval=self.model_config['voice_interval']
                                )
                
                # Dibujar UI
                self.draw_minimal_ui(annotated_frame)
                
                # Mostrar
                cv2.imshow(self.window_name, annotated_frame)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord('v'):
                    self.voice.toggle()
                elif key == ord('m'):
                    self.voice.change_gender('male')
                elif key == ord('f'):
                    self.voice.change_gender('female')
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrumpido")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpia recursos."""
        if self.camera:
            self.camera.release()
        cv2.destroyWindow(self.window_name)
        
        print(f"\n📊 Detecciones: {self.stats['detections']}")
        print(f"⏰ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
