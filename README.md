# AsistenteTIC - CASE

Asistente inteligente wearable para personas con discapacidad visual, desarrollado como TFG en el centro Salesianos ICA.

## Descripción

CASE (Computer-Aided Support Equipment) es un sistema que se ejecuta en una Raspberry Pi 4 de forma offline y proporciona:

- 🤟 **Reconocimiento de LSE** (Lengua de Signos Española) en tiempo real
- 🚦 **Detección de semáforos** peatonales mediante YOLOv8
- 👁 **Segmentación visual** con MobileSAM para describir el entorno
- 🔊 **Síntesis de voz** para comunicar las detecciones al usuario
- 📡 **Streaming de vídeo** por red local vía Flask

## Hardware

- Raspberry Pi 4 (2GB RAM)
- Cámara USB integrada en gafas wearable

## Tecnologías

- Python 3
- YOLOv8 (Ultralytics)
- MobileSAM (Meta AI)
- Flask + OpenCV
- pyttsx3 (TTS offline)

## Instalación

```bash
git clone https://github.com/Alex-SM98/AsistenteTIC.git
cd AsistenteTIC
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python server.py
```

Accede desde el navegador a `http://10.42.0.1:5000` conectado a la red WiFi **CASE-DEMO**.

## Autor

Alex Simbaña — DAM, Salesianos ICA
