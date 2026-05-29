# utils/settings_manager.py
import json
import os

class SettingsManager:
    """Gestor de configuración persistente."""
    
    SETTINGS_FILE = 'settings.json'
    
    @staticmethod
    def load():
        """Carga configuración guardada."""
        if os.path.exists(SettingsManager.SETTINGS_FILE):
            try:
                with open(SettingsManager.SETTINGS_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    @staticmethod
    def save(settings):
        """Guarda configuración."""
        try:
            with open(SettingsManager.SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"⚠️ No se pudo guardar configuración: {e}")
