"""
============================================================================
MOTOR DE VOZ MULTIPLATAFORMA
============================================================================
Soporta: macOS (say), Linux/Raspberry Pi (espeak), Windows (SAPI5)
============================================================================
"""

import pyttsx3
import subprocess
import platform
import threading
import time
from collections import defaultdict


class VoiceEngine:
    """Motor Text-to-Speech multiplataforma."""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config['enabled']
        self.gender = config.get('gender', 'male')
        self.jarvis_mode = config.get('jarvis_mode', False)
        self.last_announcement = defaultdict(float)
        self.lock = threading.Lock()
        
        # Detectar sistema operativo
        self.os_type = platform.system()
        self.use_native = (self.os_type == 'Darwin' and 
                          config.get('use_native_macos', True))
        
        # Inicializar pyttsx3 si no usamos macOS nativo
        if not self.use_native:
            try:
                self.engine = pyttsx3.init()
                self._configure_pyttsx3()
                print(f"🔊 Motor TTS: pyttsx3 ({self.os_type})")
            except Exception as e:
                print(f"⚠️ Error inicializando TTS: {e}")
                self.enabled = False
        else:
            print(f"🔊 Motor TTS: macOS native (say)")
    
    def _configure_pyttsx3(self):
        """Configura pyttsx3 según sistema operativo."""
        try:
            # Configuración básica
            rate = 140 if self.jarvis_mode else self.config.get('rate', 150)
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', self.config.get('volume', 1.0))
            
            # Seleccionar voz según OS
            voices = self.engine.getProperty('voices')
            
            if self.os_type == 'Darwin':  # macOS
                self._select_macos_voice(voices)
            elif self.os_type == 'Linux':  # Raspberry Pi
                self._select_linux_voice(voices)
            elif self.os_type == 'Windows':
                self._select_windows_voice(voices)
            else:
                # Fallback genérico
                if len(voices) > 0:
                    self.engine.setProperty('voice', voices[0].id)
        
        except Exception as e:
            print(f"⚠️ Error configurando voz: {e}")
    
    def _select_macos_voice(self, voices):
        """Selecciona voz para macOS."""
        target_names = ['Jorge', 'Diego'] if self.gender == 'male' else ['Monica', 'Paulina']
        
        for voice in voices:
            if any(name in voice.name for name in target_names):
                self.engine.setProperty('voice', voice.id)
                print(f"   ✅ Voz macOS: {voice.name}")
                return
        
        # Fallback: buscar cualquier voz en español
        for voice in voices:
            if 'es_' in str(voice.languages) or 'es-' in voice.id.lower():
                self.engine.setProperty('voice', voice.id)
                print(f"   ⚠️ Voz macOS (fallback): {voice.name}")
                return
    
    def _select_linux_voice(self, voices):
        """Selecciona voz para Linux/Raspberry Pi."""
        # En Linux, pyttsx3 usa espeak/espeak-ng
        # Las voces suelen ser: spanish, spanish-latin, es, es-la, etc.
        
        print(f"   ℹ️ Voces disponibles en Linux: {len(voices)}")
        
        # Buscar voz en español
        for voice in voices:
            voice_name = voice.name.lower()
            voice_id = voice.id.lower()
            if voice in voices:
                voice_name = voice.name.lower()
                voice_id = voice_id.lower()
                if any(x in voice_name or x in voice_id for x in
                    ['spanish', 'español', 'es-', 'es_', 'europe/es', 'es-la']):
                    self.engine.setProperty('voice', voice_id)
                    print(f"Voz linux: {voice_name} ({voice.id})")
                    return
        self.engine.setProperty('voice', 'europe/es')
        print(f"Voz forzada: europe/es")
    
    def _select_windows_voice(self, voices):
        """Selecciona voz para Windows."""
        for voice in voices:
            if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                print(f"   ✅ Voz Windows: {voice.name}")
                return
        
        # Fallback
        if len(voices) > 0:
            self.engine.setProperty('voice', voices[0].id)
            print(f"   ⚠️ Voz Windows (por defecto): {voices[0].name}")
    
    def speak(self, text, min_interval=0.0):
        """
        Pronuncia texto.
        
        Args:
            text: Texto a pronunciar
            min_interval: Tiempo mínimo entre pronunciaciones del mismo texto
        """
        if not self.enabled:
            return
        
        # Control de intervalo
        if min_interval > 0:
            current_time = time.time()
            if current_time - self.last_announcement[text] < min_interval:
                return
            self.last_announcement[text] = current_time
        
        # Pronunciar en thread separado
        threading.Thread(
            target=self._speak_blocking, 
            args=(text,), 
            daemon=True
        ).start()
    
    def _speak_blocking(self, text):
        """Pronuncia texto (bloqueante, usar en thread)."""
        with self.lock:
            try:
                if self.use_native and self.os_type == 'Darwin':
                    # macOS: comando 'say' nativo
                    voice = 'Jorge' if self.gender == 'male' else 'Monica'
                    rate = '140' if self.jarvis_mode else '150'
                    
                    result = subprocess.run(
                        ['say', '-v', voice, '-r', rate, text],
                        capture_output=True,
                        timeout=10
                    )
                    
                    # Si falla Jorge, probar con voz por defecto
                    if result.returncode != 0:
                        subprocess.run(['say', text], timeout=10)
                
                else:
                    # Linux/Windows: pyttsx3
                    self.engine.say(text)
                    self.engine.runAndWait()
            
            except FileNotFoundError:
                # Comando 'say' no existe (Linux/Windows)
                print("⚠️ Comando 'say' no disponible, usando pyttsx3")
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except:
                    pass
            
            except Exception as e:
                print(f"⚠️ Error TTS: {e}")
    
    def toggle(self):
        """Activa/desactiva voz."""
        self.enabled = not self.enabled
        return self.enabled
    
    def change_gender(self, gender):
        """Cambia género de voz."""
        self.gender = gender
        if not self.use_native:
            self._configure_pyttsx3()
    
    def toggle_jarvis_mode(self):
        """Activa/desactiva modo Jarvis."""
        self.jarvis_mode = not self.jarvis_mode
        if not self.use_native:
            self._configure_pyttsx3()
        return self.jarvis_mode
