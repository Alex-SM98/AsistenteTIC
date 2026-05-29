"""
============================================================================
COMPONENTES UI REUTILIZABLES
============================================================================
"""

import cv2
import numpy as np


class UIComponents:
    """Componentes de interfaz reutilizables."""

    @staticmethod
    def draw_rounded_rect(img, x, y, w, h, color, corner_radius=10, thickness=-1):
        """Dibuja un rectángulo con esquinas redondeadas."""
        cv2.ellipse(img, (x+corner_radius, y+corner_radius), (corner_radius, corner_radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x+w-corner_radius, y+corner_radius), (corner_radius, corner_radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x+corner_radius, y+h-corner_radius), (corner_radius, corner_radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x+w-corner_radius, y+h-corner_radius), (corner_radius, corner_radius), 0, 0, 90, color, thickness)

        if thickness > 0:
            cv2.line(img, (x+corner_radius, y), (x+w-corner_radius, y), color, thickness)
            cv2.line(img, (x+corner_radius, y+h), (x+w-corner_radius, y+h), color, thickness)
            cv2.line(img, (x, y+corner_radius), (x, y+h-corner_radius), color, thickness)
            cv2.line(img, (x+w, y+corner_radius), (x+w, y+h-corner_radius), color, thickness)
        else:
            cv2.rectangle(img, (x+corner_radius, y), (x+w-corner_radius, y+h), color, -1)
            cv2.rectangle(img, (x, y+corner_radius), (x+w, y+h-corner_radius), color, -1)

    @staticmethod
    def draw_button(img, x, y, w, h, text, bg_color, text_color, hover=False,
                    corner_radius=8, font_scale=0.7, icon=None):
        """Dibuja un botón elegante con efecto hover."""

        # Paleta blanco/negro — ignora bg_color externo
        if hover:
            btn_color = (255, 255, 255)   # blanco al hacer hover
            txt_color = (0, 0, 0)         # texto negro sobre blanco
            border_color = (255, 255, 255)
            shadow_offset = 6
        else:
            btn_color = (30, 30, 30)      # gris muy oscuro (casi negro)
            txt_color = (255, 255, 255)   # texto blanco sobre oscuro
            border_color = (200, 200, 200) # borde gris claro
            shadow_offset = 4

        # Sombra
        UIComponents.draw_rounded_rect(img, x+shadow_offset, y+shadow_offset, w, h,
                                       (0, 0, 0), corner_radius, -1)

        # Fondo del botón
        UIComponents.draw_rounded_rect(img, x, y, w, h, btn_color, corner_radius, -1)

        # Borde
        UIComponents.draw_rounded_rect(img, x, y, w, h, border_color, corner_radius, 2)

        # Texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2

        # Sombra del texto
        cv2.putText(img, text, (text_x+1, text_y+1), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, txt_color, thickness)

    @staticmethod
    def draw_card(img, x, y, w, h, title, content, color, corner_radius=12):
        """Dibuja una tarjeta informativa elegante."""

        # Sombra
        UIComponents.draw_rounded_rect(img, x+5, y+5, w, h, (0, 0, 0), corner_radius, -1)

        # Fondo oscuro (negro/gris muy oscuro)
        UIComponents.draw_rounded_rect(img, x, y, w, h, (25, 25, 25), corner_radius, -1)

        # Borde blanco (en lugar de color)
        UIComponents.draw_rounded_rect(img, x, y, w, h, (200, 200, 200), corner_radius, 2)

        # Barra superior blanca
        cv2.rectangle(img, (x+corner_radius, y), (x+w-corner_radius, y+4), (255, 255, 255), -1)

        # Título en gris claro
        if title:
            cv2.putText(img, title, (x+15, y+30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (180, 180, 180), 1, cv2.LINE_AA)

        # Contenido en blanco
        if content:
            cv2.putText(img, str(content), (x+15, y+70), cv2.FONT_HERSHEY_DUPLEX,
                       1.8, (255, 255, 255), 3, cv2.LINE_AA)

    @staticmethod
    def draw_status_indicator(img, x, y, active, size=8):
        """Dibuja un indicador de estado (LED)."""
        # Activo: blanco brillante | Inactivo: gris oscuro
        color = (255, 255, 255) if active else (80, 80, 80)
        cv2.circle(img, (x, y), size+2, (0, 0, 0), -1)
        cv2.circle(img, (x, y), size, color, -1)
        if active:
            cv2.circle(img, (x-2, y-2), size//3, (200, 200, 200), -1)
