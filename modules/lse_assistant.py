"""
============================================================================
INTÉRPRETE LSE
============================================================================
Reconocimiento de Lengua de Signos Española con interfaz moderna
============================================================================
"""


import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque, Counter
from utils import CameraManager, VoiceEngine, UIComponents



class LSEAssistant:
    """Intérprete de Lengua de Signos Española."""
    
    def __init__(self, model_config, camera_config, voice_config):
        self.model_config = model_config
        self.camera_config = camera_config
        self.voice_config = voice_config
        self.frame_skip = 0
        self.last_annotated_frame = None
        
        self.model = None
        self.voice = None
        self.camera = None
        
        self.prediction_buffer = deque(maxlen=model_config['buffer_size'])
        self.accumulated_text = ""
        self.last_spoken_letter = ""
        self.last_spoken_time = 0
        
        self.current_letter = None
        self.current_confidence = 0.0
        
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        
        self.mouse_x = 0
        self.mouse_y = 0
        self.window_name = "LSE Interpreter - AsistenteTIC"
        
        self.buttons = {
            'speak':      {'rect': (0, 0, 0, 0), 'hover': False},
            'clear':      {'rect': (0, 0, 0, 0), 'hover': False},
            'conf_up':    {'rect': (0, 0, 0, 0), 'hover': False},
            'conf_down':  {'rect': (0, 0, 0, 0), 'hover': False},
            'exit':       {'rect': (0, 0, 0, 0), 'hover': False},
        }
        
        self._init_model()
        self._init_voice()
    
    def _init_model(self):
        """Inicializa el modelo YOLO."""
        print(f"📦 Cargando modelo LSE...")
        try:
            self.model = YOLO(self.model_config['path'])
            print(f"   ✅ Modelo cargado: {self.model_config['classes']} letras")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            raise
    
    def _init_voice(self):
        """Inicializa el motor de voz."""
        self.voice = VoiceEngine(self.voice_config)
    
    def process_frame(self, frame):
        """Procesa un frame con YOLO."""
        self.frame_skip += 1
        if self.frame_skip % 2 != 0:
            return self.last_annotated_frame if hasattr(self, 'last_annotated_frame') else frame

        results = self.model(
            frame,
            verbose=False,
            conf=self.model_config['conf'],
            iou=self.model_config['iou'],
            max_det=self.model_config['max_det'],
            agnostic_nms=True
        )
        
        annotated_frame = results[0].plot(line_width=2, font_size=1)
        
        self.current_letter = None
        self.current_confidence = 0.0
        
        if len(results[0].boxes) > 0:
            best_idx = results[0].boxes.conf.argmax()
            best_box = results[0].boxes[best_idx]
            
            class_id = int(best_box.cls[0])
            self.current_confidence = float(best_box.conf[0])
            self.current_letter = self.model.names[class_id]

            min_conf = self.model_config['conf']

            for box in results[0].boxes:
                cid = int(box.cls[0])
                cname = self.model.names[cid]
                cconf = float(box.conf[0])
                print("Det:", cname, f"{cconf:.2f}")

            if self.current_confidence < min_conf:
                self.current_letter = None
                self.current_confidence = 0.0
                return annotated_frame
            
            self.prediction_buffer.append(self.current_letter)
            
            if len(self.prediction_buffer) >= self.model_config['buffer_size']:
                stable_letter, count = Counter(self.prediction_buffer).most_common(1)[0]
                required = int(self.model_config['buffer_size'] * 0.7)

                if count >= required:
                    current_time = time.time()
                    if (stable_letter != self.last_spoken_letter and
                        current_time - self.last_spoken_time > self.model_config['voice_interval']):

                        self.voice.speak(stable_letter, min_interval=0)
                        self.last_spoken_letter = stable_letter
                        self.last_spoken_time = current_time
                        self.accumulated_text += stable_letter
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.fps = 30 / (time.time() - self.fps_start_time)
            self.fps_start_time = time.time()
        
        self.last_annotated_frame = annotated_frame
        return annotated_frame
    
    def draw_ui(self, frame):
        """Dibuja interfaz en blanco y negro sobre el frame."""
        h, w = frame.shape[:2]

        # ── Panel lateral ──────────────────────────────────────────────────
        panel_width = 380
        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
        panel[:] = (245, 245, 245)   # fondo blanco/gris muy claro

        y = 20

        # ── HEADER ────────────────────────────────────────────────────────
        cv2.putText(panel, "LSE", (20, y+35),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(panel, "INTERPRETER", (20, y+70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.line(panel, (20, y+85), (panel_width-20, y+85), (180, 180, 180), 2)

        y += 110

        # ── LETRA DETECTADA ───────────────────────────────────────────────
        # Borde: blanco si hay letra, gris si no
        card_border = (60, 60, 60) if self.current_letter else (180, 180, 180)
        UIComponents.draw_card(panel, 20, y, panel_width-40, 130,
                               "LETRA DETECTADA",
                               self.current_letter if self.current_letter else "---",
                               card_border)

        if self.current_letter:
            conf_text = f"{self.current_confidence*100:.0f}%"
            cv2.putText(panel, conf_text, (45, y+110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1, cv2.LINE_AA)

        y += 150

        # ── PALABRA ACUMULADA ─────────────────────────────────────────────
        UIComponents.draw_card(panel, 20, y, panel_width-40, 100,
                               "PALABRA", None, (120, 120, 120))

        word_display = self.accumulated_text if self.accumulated_text else "---"
        if len(word_display) > 14:
            word_display = "..." + word_display[-14:]

        cv2.putText(panel, word_display, (35, y+70),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        y += 120

        # ── AJUSTE DE CONFIANZA ───────────────────────────────────────────
        conf_y = y
        cv2.putText(panel, "CONFIANZA", (20, conf_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        conf_y += 25

        conf_value = f"{self.model_config['conf']:.2f}"
        cv2.putText(panel, conf_value, (160, conf_y+30),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)

        btn_size = 45

        # Botón  −  (gris oscuro)
        btn_x = 20
        btn_y = conf_y
        self.buttons['conf_down']['rect'] = (btn_x + w, btn_y, btn_size, btn_size)
        UIComponents.draw_button(panel, btn_x, btn_y, btn_size, btn_size,
                                 "-", (60, 60, 60), (255, 255, 255),
                                 self.buttons['conf_down']['hover'],
                                 corner_radius=8, font_scale=1.2)

        # Botón  +  (gris oscuro)
        btn_x = panel_width - 20 - btn_size
        self.buttons['conf_up']['rect'] = (btn_x + w, btn_y, btn_size, btn_size)
        UIComponents.draw_button(panel, btn_x, btn_y, btn_size, btn_size,
                                 "+", (60, 60, 60), (255, 255, 255),
                                 self.buttons['conf_up']['hover'],
                                 corner_radius=8, font_scale=1.2)

        y += 80

        # ── BOTONES PRINCIPALES ───────────────────────────────────────────
        btn_width  = panel_width - 40
        btn_height = 55
        btn_spacing = 15

        # PRONUNCIAR PALABRA
        self.buttons['speak']['rect'] = (20 + w, y, btn_width, btn_height)
        UIComponents.draw_button(panel, 20, y, btn_width, btn_height,
                                 "PRONUNCIAR PALABRA",
                                 (40, 40, 40), (255, 255, 255),
                                 self.buttons['speak']['hover'])
        y += btn_height + btn_spacing

        # LIMPIAR TEXTO
        self.buttons['clear']['rect'] = (20 + w, y, btn_width, btn_height)
        UIComponents.draw_button(panel, 20, y, btn_width, btn_height,
                                 "LIMPIAR TEXTO",
                                 (40, 40, 40), (255, 255, 255),
                                 self.buttons['clear']['hover'])
        y += btn_height + btn_spacing

        # VOLVER AL MENÚ
        self.buttons['exit']['rect'] = (20 + w, y, btn_width, btn_height)
        UIComponents.draw_button(panel, 20, y, btn_width, btn_height,
                                 "VOLVER AL MENU",
                                 (20, 20, 20), (255, 255, 255),
                                 self.buttons['exit']['hover'])

        # ── FOOTER ────────────────────────────────────────────────────────
        y = h - 50

        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(panel, fps_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        voice_x = panel_width - 60
        UIComponents.draw_status_indicator(panel, voice_x, y-5, self.voice.enabled)
        cv2.putText(panel, "VOZ", (voice_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

        combined = np.hstack([frame, panel])
        return combined
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback del ratón."""
        self.mouse_x = x
        self.mouse_y = y
        
        for key, btn in self.buttons.items():
            bx, by, bw, bh = btn['rect']
            btn['hover'] = (bx <= x <= bx + bw and by <= y <= by + bh)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.buttons['speak']['hover']:
                self.speak_word()
            elif self.buttons['clear']['hover']:
                self.clear_text()
            elif self.buttons['conf_up']['hover']:
                self.adjust_confidence(0.05)
            elif self.buttons['conf_down']['hover']:
                self.adjust_confidence(-0.05)
            elif self.buttons['exit']['hover']:
                self.exit_requested = True
    
    def speak_word(self):
        """Pronuncia la palabra acumulada."""
        if self.accumulated_text:
            self.voice.speak(self.accumulated_text, min_interval=2)
            print(f"🗣️ Pronunciando: {self.accumulated_text}")
        else:
            print("⚠️ No hay texto para pronunciar")
    
    def clear_text(self):
        """Limpia el texto acumulado."""
        self.accumulated_text = ""
        self.prediction_buffer.clear()
        self.last_spoken_letter = ""
        print("🗑️ Texto limpiado")
    
    def adjust_confidence(self, delta):
        """Ajusta el umbral de confianza."""
        self.model_config['conf'] = max(0.1, min(0.95,
                                                  self.model_config['conf'] + delta))
        print(f"📊 Confianza: {self.model_config['conf']:.2f}")
    
    def run(self):
        """Ejecuta el intérprete LSE."""
        print("\n" + "=" * 70)
        print("🤟 INTÉRPRETE LSE - VERSIÓN CLEAN")
        print("=" * 70)
        
        self.exit_requested = False
        
        try:
            self.camera = CameraManager(
                self.camera_config['id'],
                self.camera_config['width'],
                self.camera_config['height']
            )
            
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            
            while not self.exit_requested:
                frame = self.camera.read_frame()
                if frame is None:
                    break
                
                processed_frame = self.process_frame(frame)
                final_frame = self.draw_ui(processed_frame)
                
                cv2.imshow(self.window_name, final_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord('v'):
                    self.voice.toggle()
                    print(f"🔊 Voz: {'ON' if self.voice.enabled else 'OFF'}")
                elif key == ord('p'):
                    self.speak_word()
                elif key == ord('c'):
                    self.clear_text()
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrumpido")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpia recursos."""
        if self.camera:
            self.camera.release()
        cv2.destroyWindow(self.window_name)
        
        print("\n✅ Intérprete LSE cerrado")
        if self.accumulated_text:
            print(f"📝 Texto final: {self.accumulated_text}")
