"""
============================================================================
ASISTENTE TIC - MENÚ PRINCIPAL
============================================================================
Sistema integrado de dos asistentes:
1. Interprete de Lengua de Signos Española (LSE)
2. Asistente Visual para personas con discapacidad visual

Proyecto: TFG - Premio Don Bosco de Zaragoza
Autor: Alex Simbaña
============================================================================
"""

import cv2
import numpy as np
import sys
import argparse
from config import MODELS, CAMERA, VOICE, COLORS, MESSAGES
from utils import VoiceEngine, UIComponents
from modules import LSEAssistant, VisualAssistant


class AsistenteTICMenu:
    """Menú principal de selección de asistente."""
    
    def __init__(self):
        self.window_name = "AsistenteTIC - Seleccion"
        self.selected_option = None
        self.mouse_pos = (0, 0)
        self.running_assistant = None
        self.show_menu = True
        
        # Botones (coordenadas pensadas para 1080x600)
        self.buttons = {
            'lse':    {'rect': (120, 190, 360, 150), 'hover': False},
            'visual': {'rect': (600, 190, 360, 150), 'hover': False},
            'exit':   {'rect': (340, 420, 400, 70),  'hover': False},
        }
        
        self.voice = VoiceEngine(VOICE)
    
    def speak_jarvis(self, message, use_jarvis_voice=True):
        """Mensajes tipo Jarvis con delay mínimo."""
        if use_jarvis_voice:
            jarvis_messages = {
                'startup': 'Bienvenido al Asistente TIC. Sistema KEYS iniciado. Por favor seleccione un modelo.',
                'shutdown': 'Sistema KEYS finalizado. Hasta pronto.',
                'lse_startup': 'Activando modulo de interpretacion de Lengua de Signos.',
                'visual_startup': 'Activando modulo de asistencia visual para deteccion de semaforos peatonales.',
                'return_menu': 'Volviendo al menu principal.',
                'error': 'Error detectado. Verifique la configuracion del sistema.',
            }
            text = jarvis_messages.get(message, message)
        else:
            text = MESSAGES.get(message, message)
        
        self.voice.speak(text, min_interval=2.5)
    
    def draw_menu(self):
        """Dibuja el menú principal (fondo blanco, diseño limpio)."""
        canvas = np.ones((600, 1080, 3), dtype=np.uint8) * 255

        # Título centrado
        font = cv2.FONT_HERSHEY_SIMPLEX

        title = "ASISTENTE TIC  -  CASE"
        title_scale = 0.9
        title_thick = 2
        title_size = cv2.getTextSize(title, font, title_scale, title_thick)[0]
        title_x = (1080 - title_size[0]) // 2
        title_y = 50
        cv2.putText(canvas, title, (title_x, title_y),
            font, title_scale, (0, 0, 0), title_thick, cv2.LINE_AA)

        subtitle = "Asistente visual y de lengua de signos en tiempo real"
        sub_scale = 0.55
        sub_thick = 1
        sub_size = cv2.getTextSize(subtitle, font, sub_scale, sub_thick)[0]
        sub_x = (1080 - sub_size[0]) // 2
        sub_y = 80
        cv2.putText(canvas, subtitle, (sub_x, sub_y),
            font, sub_scale, (80, 80, 80), sub_thick, cv2.LINE_AA)


        # Línea fina
        cv2.line(canvas, (220, 110), (860, 110), (210, 210, 210), 1)

        # Tarjeta LSE
        self._draw_card(
            canvas,
            *self.buttons['lse']['rect'],
            "Interprete LSE",
            "Usuario: persona sorda",    
            "Tecla: 1 o L",
            self.buttons['lse']['hover']
        )

        # Tarjeta Visual
        self._draw_card(
            canvas,
            *self.buttons['visual']['rect'],
            "Asistente Visual",
            "Usuario: persona ciega",    
            "Tecla: 2 o V",
            self.buttons['visual']['hover']
        )

        # Botón Salir
        self._draw_button(
            canvas,
            *self.buttons['exit']['rect'],
            "SALIR (Q / ESC)",
            COLORS['danger'],
            self.buttons['exit']['hover']
        )

        # Footer con autor
        footer_y = 565
        font = cv2.FONT_HERSHEY_SIMPLEX

        footer1 = "Proyecto: TFG - Premio Don Bosco de Zaragoza  |  Autor: Alex Simbana"
        size1 = cv2.getTextSize(footer1, font, 0.48, 1)[0]
        x1 = (1080 - size1[0]) // 2
        cv2.putText(canvas, footer1, (x1, footer_y),
            font, 0.48, (60, 60, 60), 1, cv2.LINE_AA)

        footer2 = "Controles: 1=LSE   2=Visual   Q/Esc=Salir   M=Menu"
        size2 = cv2.getTextSize(footer2, font, 0.48, 1)[0]
        x2 = (1080 - size2[0]) // 2
        cv2.putText(canvas, footer2, (x2, footer_y + 18),
            font, 0.48, (100, 100, 100), 1, cv2.LINE_AA)
        return canvas
    
    def _draw_card(self, canvas, x, y, w, h, title, line1, line2, hover):
        """Tarjeta minimalista con dos lineas centradas."""
        if hover:
            shadow_offset = 3
        else:
            shadow_offset = 1

        # Sombra ligera
        UIComponents.draw_rounded_rect(canvas, x + shadow_offset, y + shadow_offset,
                                   w, h, (225, 225, 225), 18, -1)
        # Fondo blanco + borde negro
        UIComponents.draw_rounded_rect(canvas, x, y, w, h, (255, 255, 255), 18, -1)
        UIComponents.draw_rounded_rect(canvas, x, y, w, h, (0, 0, 0), 18, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Título centrado
        font_scale_title = 0.9
        thickness_title = 2
        title_size = cv2.getTextSize(title, font, font_scale_title, thickness_title)[0]
        title_x = x + (w - title_size[0]) // 2
        title_y = y + 70
        cv2.putText(canvas, title, (title_x, title_y),
                font, font_scale_title, (0, 0, 0), thickness_title, cv2.LINE_AA)

        # Linea 1
        font_scale_sub = 0.5
        thickness_sub = 1
        line1_size = cv2.getTextSize(line1, font, font_scale_sub, thickness_sub)[0]
        line1_x = x + (w - line1_size[0]) // 2
        line1_y = title_y + 25
        cv2.putText(canvas, line1, (line1_x, line1_y),
                font, font_scale_sub, (60, 60, 60), thickness_sub, cv2.LINE_AA)

        # Linea 2
        line2_size = cv2.getTextSize(line2, font, font_scale_sub, thickness_sub)[0]
        line2_x = x + (w - line2_size[0]) // 2
        line2_y = line1_y + 20
        cv2.putText(canvas, line2, (line2_x, line2_y),
                font, font_scale_sub, (60, 60, 60), thickness_sub, cv2.LINE_AA)


    
    def _draw_button(self, canvas, x, y, w, h, text, color, hover):
        """Botón principal negro, centrado."""
        bg = (0, 0, 0)
        if hover:
            bg = (30, 30, 30)
        UIComponents.draw_button(canvas, x, y, w, h, text, bg,
                                 (255, 255, 255), hover,
                                 corner_radius=22, font_scale=0.9)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback del ratón."""
        self.mouse_pos = (x, y)
        for key, btn in self.buttons.items():
            bx, by, bw, bh = btn['rect']
            btn['hover'] = (bx <= x <= bx + bw and by <= y <= by + bh)
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, btn in self.buttons.items():
                if btn['hover']:
                    self.selected_option = key
    
    def handle_keyboard(self, key):
        """Atajos de teclado globales."""
        if key == ord('q') or key == 27:
            return 'exit'
        if key in (ord('1'), ord('l'), ord('L')):
            return 'lse'
        if key in (ord('2'), ord('v'), ord('V')):
            return 'visual'
        if key in (ord('m'), ord('M')):
            return 'menu'
        return None
    
    def run(self):
        """Ejecuta el menú."""
        print("=" * 70)
        print("🎯 ASISTENTE TIC - MENÚ PRINCIPAL")
        print("=" * 70)
        print("Proyecto: TFG - Premio Don Bosco")
        print("Autor: Alex Simbaña")
        print("=" * 70)
        print("\n💡 Atajos de teclado:")
        print("   1 o L: Interprete LSE")
        print("   2 o V: Asistente Visual")
        print("   Q o ESC: Salir")
        print("   M: Volver al menu\n")
        
        self.speak_jarvis('startup')
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            if self.show_menu:
                canvas = self.draw_menu()
                cv2.imshow(self.window_name, canvas)
            
            key = cv2.waitKey(1) & 0xFF
            action = self.handle_keyboard(key)
            
            if self.selected_option:
                action = self.selected_option
                self.selected_option = None
            
            if action == 'exit':
                break
            elif action == 'lse':
                cv2.destroyWindow(self.window_name)
                self.show_menu = False
                self._launch_lse()
                self.show_menu = True
                cv2.namedWindow(self.window_name)
                cv2.setMouseCallback(self.window_name, self.mouse_callback)
            elif action == 'visual':
                cv2.destroyWindow(self.window_name)
                self.show_menu = False
                self._launch_visual()
                self.show_menu = True
                cv2.namedWindow(self.window_name)
                cv2.setMouseCallback(self.window_name, self.mouse_callback)
            elif action == 'menu':
                if not self.show_menu:
                    self.speak_jarvis('return_menu')
                    self.show_menu = True
        
        cv2.destroyAllWindows()
        self.speak_jarvis('shutdown')
        print("\n✅ Aplicación cerrada")
    
    def _launch_lse(self):
        print("\n🤟 Iniciando Interprete LSE...")
        self.speak_jarvis('lse_startup')
        assistant = LSEAssistant(MODELS['lse'], CAMERA, VOICE)
        assistant.run()
        self.speak_jarvis('return_menu')
    
    def _launch_visual(self):
        print("\n👁️ Iniciando Asistente Visual...")
        self.speak_jarvis('visual_startup')
        assistant = VisualAssistant(MODELS['visual'], CAMERA, VOICE)
        assistant.run()
        self.speak_jarvis('return_menu')


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(description='AsistenteTIC - Sistema CASE')
    parser.add_argument('--mode', choices=['lse', 'visual', 'menu'], default='menu',
                        help='Modo de inicio directo (default: menu)')
    parser.add_argument('--no-voice', action='store_true',
                        help='Desactivar mensajes de voz del sistema')
    args = parser.parse_args()
    
    if args.no_voice:
        VOICE['enabled'] = False
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                    ASISTENTE TIC - CASE                    ║
    ║          Sistema Integrado de Asistencia Tecnologica       ║
    ║                                                            ║
    ║  Proyecto: TFG           
    ║  Autor: Alex Simbaña                                       ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    try:
        if args.mode == 'lse':
            print("🚀 Modo directo: Interprete LSE")
            voice = VoiceEngine(VOICE)
            voice.speak("Activando interprete de Lengua de Signos Espanola", min_interval=2.5)
            assistant = LSEAssistant(MODELS['lse'], CAMERA, VOICE)
            assistant.run()
        elif args.mode == 'visual':
            print("🚀 Modo directo: Asistente Visual")
            voice = VoiceEngine(VOICE)
            voice.speak("Activando asistente visual para deteccion de semaforos", min_interval=2.5)
            assistant = VisualAssistant(MODELS['visual'], CAMERA, VOICE)
            assistant.run()
        else:
            menu = AsistenteTICMenu()
            menu.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
