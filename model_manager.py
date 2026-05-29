"""
============================================================================
GESTOR DE MODELOS
============================================================================
Gestiona la carga y descarga de modelos YOLO en memoria.
Solo un modelo activo a la vez para no saturar la RAM de la Pi.
============================================================================
"""

from ultralytics import YOLO
import gc


class ModelManager:
    
    def __init__(self, model_paths):
        # Rutas de los modelos disponibles
        # Ej: {'lse': 'models/lse_best.pt', 'semaforos': 'models/semaforos_best.pt'}
        self.model_paths = model_paths
        
        self.current_model = None      # instancia YOLO activa
        self.current_mode = None       # nombre del modo activo ('lse', 'semaforos', None)
    
    def load(self, mode):
        """Carga el modelo del modo indicado. Descarga el anterior si hay uno activo."""
        
        if mode == self.current_mode:
            # El modelo pedido ya está cargado, no hacemos nada
            return True
        
        if mode not in self.model_paths:
            print(f"❌ Modo desconocido: {mode}")
            return False
        
        # Descargar modelo anterior de memoria
        self._unload()
        
        # Cargar nuevo modelo
        print(f"📦 Cargando modelo: {mode}...")
        try:
            self.current_model = YOLO(self.model_paths[mode])
            self.current_mode = mode
            print(f"   ✅ Modelo '{mode}' listo")
            return True
        except Exception as e:
            print(f"   ❌ Error cargando modelo: {e}")
            self.current_model = None
            self.current_mode = None
            return False
    
    def _unload(self):
        """Descarga el modelo actual de memoria RAM."""
        if self.current_model is not None:
            print(f"🗑️  Descargando modelo: {self.current_mode}")
            del self.current_model
            self.current_model = None
            self.current_mode = None
            gc.collect()  # forzar liberación de memoria
    
    def stop(self):
        """Para el procesamiento y descarga el modelo activo."""
        self._unload()
        print("⏹️  Sistema detenido")
    
    def predict(self, frame, conf=0.5, iou=0.45, imgsz=320):
        """Ejecuta inferencia sobre un frame. Devuelve None si no hay modelo activo."""
        if self.current_model is None:
            return None
        
        results = self.current_model(
            frame,
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=imgsz,       # resolución interna de inferencia (320 = más rápido en Pi)
            agnostic_nms=True
        )
        return results
    
    @property
    def is_active(self):
        """True si hay un modelo cargado y activo."""
        return self.current_model is not None
    
    @property
    def mode(self):
        """Devuelve el modo activo actual."""
        return self.current_mode
