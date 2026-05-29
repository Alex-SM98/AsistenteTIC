"""
============================================================================
SERVIDOR FLASK - CASE
============================================================================
Servidor principal que:
  - Gestiona la cámara en la Raspberry Pi
  - Ejecuta los modelos YOLO
  - Hace streaming de vídeo por red local (MJPEG)
  - Expone endpoints HTTP para controlar el modo desde el Mac

Acceso desde el Mac: http://<IP_RASPBERRY>:5000
============================================================================
"""

from flask import Flask, Response, render_template, jsonify, request
import cv2
import threading
import time
from collections import deque, Counter
from model_manager import ModelManager
from utils import VoiceEngine
from config import MODELS, CAMERA, VOICE, MESSAGES
LSE_MODEL_CONFIG = MODELS['lse']
VISUAL_MODEL_CONFIG = MODELS['visual']
CAMERA_CONFIG = CAMERA
VOICE_CONFIG = VOICE


app = Flask(__name__)

# ─────────────────────────────────────────────
# ESTADO GLOBAL DEL SISTEMA
# Usamos un objeto compartido entre el hilo de
# captura y los endpoints HTTP de Flask
# ─────────────────────────────────────────────
state = {
    'mode': None,           # 'lse', 'semaforos' o None
    'fps': 0.0,
    'letter': None,         # última letra detectada (LSE)
    'confidence': 0.0,
    'accumulated_text': '', # texto acumulado en modo LSE
    'detection': None,      # última detección (semáforos)
    'running': True,        # False para detener el hilo
}

# Lock para acceso seguro al estado desde múltiples hilos
state_lock = threading.Lock()

# Frame actual para el streaming (bytes JPEG)
output_frame = None
frame_lock = threading.Lock()

# Gestor de modelos
manager = ModelManager({
    'lse':       LSE_MODEL_CONFIG['path'],
    'semaforos': VISUAL_MODEL_CONFIG['path'],
})

# Motor de voz
voice = VoiceEngine(VOICE_CONFIG)

# Buffer para estabilizar predicciones LSE
prediction_buffer = deque(maxlen=LSE_MODEL_CONFIG['buffer_size'])
last_spoken_letter = ""
last_spoken_time = 0


# ─────────────────────────────────────────────
# HILO DE CAPTURA Y PROCESAMIENTO
# Corre en segundo plano, independiente de Flask
# ─────────────────────────────────────────────
def capture_loop():
    """
    Hilo principal de captura.
    Lee frames de la cámara, aplica el modelo activo,
    codifica el frame como JPEG y lo deja disponible
    para el streaming.
    """
    global output_frame, last_spoken_letter, last_spoken_time

    # Abrir cámara
    cap = cv2.VideoCapture('/dev/video0')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_CONFIG['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['height'])
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps_counter = 0
    fps_start = time.time()
    frame_skip = 0

    print("📹 Hilo de captura iniciado")

    while state['running']:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ── Cálculo de FPS ──────────────────────────────
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            with state_lock:
                state['fps'] = round(fps_counter / (time.time() - fps_start), 1)
            fps_counter = 0
            fps_start = time.time()

        # ── Frame skip (1 de cada 2 frames con IA) ──────
        frame_skip += 1
        annotated = frame.copy()

        if frame_skip % 2 == 0 and manager.is_active:
            results = manager.predict(
                frame,
                conf=LSE_MODEL_CONFIG['conf'],
                iou=LSE_MODEL_CONFIG['iou'],
                imgsz=LSE_MODEL_CONFIG.get('imgsz', 320)
            )

            if results:
                annotated = results[0].plot(line_width=2)
                _process_results(results)

        # ── Overlay de información sobre el frame ───────
        _draw_overlay(annotated)

        # ── Codificar frame como JPEG para el stream ────
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            output_frame = buffer.tobytes()

    cap.release()
    print("📹 Hilo de captura detenido")


def _process_results(results):
    """Procesa las detecciones según el modo activo."""
    global last_spoken_letter, last_spoken_time

    mode = manager.mode

    if mode == 'lse':
        # ── Modo LSE: acumular letras estables ──────────
        if len(results[0].boxes) > 0:
            best = results[0].boxes[results[0].boxes.conf.argmax()]
            letter = manager.current_model.names[int(best.cls[0])]
            conf = float(best.conf[0])

            with state_lock:
                state['letter'] = letter
                state['confidence'] = conf

            prediction_buffer.append(letter)

            # Letra estable si aparece en >70% del buffer
            if len(prediction_buffer) >= LSE_MODEL_CONFIG['buffer_size']:
                stable, count = Counter(prediction_buffer).most_common(1)[0]
                required = int(LSE_MODEL_CONFIG['buffer_size'] * 0.7)

                if count >= required:
                    now = time.time()
                    if (stable != last_spoken_letter and
                            now - last_spoken_time > LSE_MODEL_CONFIG['voice_interval']):
                        voice.speak(stable, min_interval=0)
                        last_spoken_letter = stable
                        last_spoken_time = now
                        with state_lock:
                            state['accumulated_text'] += stable
        else:
            with state_lock:
                state['letter'] = None
                state['confidence'] = 0.0

    elif mode == 'semaforos':
        # ── Modo semáforos: anunciar rojo/verde ─────────
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_name = manager.current_model.names[int(box.cls[0])]
                if float(box.conf[0]) > VISUAL_MODEL_CONFIG['conf']:
                    if class_name in ['red', 'green']:
                        voice.speak(
                            MESSAGES[class_name],
                            min_interval=VISUAL_MODEL_CONFIG['voice_interval']
                        )
                        with state_lock:
                            state['detection'] = class_name


def _draw_overlay(frame):
    """Dibuja información básica sobre el frame (modo, FPS, detección)."""
    h, w = frame.shape[:2]

    # Fondo semitransparente arriba
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Modo activo
    mode_text = f"MODO: {manager.mode.upper() if manager.mode else 'STANDBY'}"
    cv2.putText(frame, mode_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {state['fps']}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Letra detectada (modo LSE)
    if manager.mode == 'lse' and state['letter']:
        cv2.putText(frame, f"Letra: {state['letter']}  {state['confidence']*100:.0f}%",
                    (w - 220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ─────────────────────────────────────────────
# GENERADOR DE STREAMING MJPEG
# Flask llama a esta función continuamente para
# enviar los frames al navegador del Mac
# ─────────────────────────────────────────────
def generate_stream():
    """Generador que envía frames JPEG al navegador en formato MJPEG."""
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.05)
                continue
            frame = output_frame

        # Formato MJPEG: cada frame va precedido de headers HTTP
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)   # ~30 fps máximo al cliente


# ─────────────────────────────────────────────
# ENDPOINTS HTTP
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Sirve el panel de control HTML al navegador del Mac."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Endpoint de streaming de vídeo MJPEG."""
    return Response(
        generate_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/set_mode', methods=['POST'])
def set_mode():
    """
    Cambia el modo activo (lse / semaforos / stop).
    El Mac envía: POST /set_mode   Body: {"mode": "lse"}
    """
    data = request.get_json()
    mode = data.get('mode')

    if mode == 'stop':
        manager.stop()
        prediction_buffer.clear()
        with state_lock:
            state['mode'] = None
            state['letter'] = None
            state['accumulated_text'] = ''
            state['detection'] = None
        return jsonify({'status': 'ok', 'mode': 'stop'})

    success = manager.load(mode)
    if success:
        prediction_buffer.clear()
        with state_lock:
            state['mode'] = mode
            state['accumulated_text'] = ''
        return jsonify({'status': 'ok', 'mode': mode})
    else:
        return jsonify({'status': 'error', 'message': f'No se pudo cargar {mode}'}), 500


@app.route('/status')
def status():
    """Devuelve el estado actual del sistema en JSON. El panel HTML lo consulta cada segundo."""
    with state_lock:
        return jsonify({
            'mode':             state['mode'],
            'fps':              state['fps'],
            'letter':           state['letter'],
            'confidence':       round(state['confidence'] * 100),
            'accumulated_text': state['accumulated_text'],
            'detection':        state['detection'],
        })


@app.route('/clear_text', methods=['POST'])
def clear_text():
    """Limpia el texto acumulado en modo LSE."""
    with state_lock:
        state['accumulated_text'] = ''
    prediction_buffer.clear()
    return jsonify({'status': 'ok'})


@app.route('/toggle_voice', methods=['POST'])
def toggle_voice():
    """Activa o desactiva el motor de voz."""
    voice.toggle()
    return jsonify({'status': 'ok', 'voice': voice.enabled})


# ─────────────────────────────────────────────
# ARRANQUE
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Iniciar hilo de captura en segundo plano
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    print("\n" + "=" * 60)
    print("🌐 CASE - Servidor iniciado")
    print("=" * 60)
    print("   Accede desde el Mac a:")
    print("   http://<IP_RASPBERRY>:5000")
    print("   (Hotspot propio: http://10.42.0.1:5000)")
    print("=" * 60 + "\n")

    # host='0.0.0.0' para aceptar conexiones desde cualquier IP de la red
    app.run(host='0.0.0.0', port=5000, threaded=True)
