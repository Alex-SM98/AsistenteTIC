"""
============================================================================
GESTOR DE CÁMARA ROBUSTO
============================================================================
Auto-detección de cámaras disponibles y fallback automático
============================================================================
"""

import cv2


class CameraManager:
    """Gestiona el acceso a la cámara con detección automática."""
    
    def __init__(self, camera_id, width, height):
        self.requested_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.actual_id = None
        self.actual_width = None
        self.actual_height = None
        self._init_camera()
    
    def _detect_available_cameras(self, max_cameras=5):
        """
        Detecta cámaras disponibles en el sistema.
        
        Returns:
            list: Lista de IDs de cámaras disponibles
        """
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
    
    def _test_camera(self, camera_id):
        """
        Prueba si una cámara funciona correctamente.
        
        Args:
            camera_id: ID de la cámara a probar
            
        Returns:
            tuple: (bool éxito, VideoCapture object o None)
        """
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                return False, None
            
            # Intentar leer un frame de prueba
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return False, None
            
            # Cámara funciona
            return True, cap
            
        except Exception as e:
            print(f"   ⚠️ Error probando cámara {camera_id}: {e}")
            return False, None
    
    def _init_camera(self):
        """Inicializa la cámara con detección automática y fallback."""
        print(f"📹 Iniciando sistema de cámara...")
        
        # 1. Detectar cámaras disponibles
        available_cameras = self._detect_available_cameras()
        
        if not available_cameras:
            raise RuntimeError("❌ No se detectaron cámaras en el sistema")
        
        print(f"   ℹ️ Cámaras detectadas: {available_cameras}")
        
        # 2. Intentar con la cámara solicitada primero
        if self.requested_id in available_cameras:
            print(f"   🔍 Intentando cámara {self.requested_id}...")
            success, cap = self._test_camera(self.requested_id)
            
            if success:
                self.cap = cap
                self.actual_id = self.requested_id
                self._configure_camera()
                return
            else:
                print(f"   ⚠️ Cámara {self.requested_id} no respondió correctamente")
        
        # 3. Fallback: probar con otras cámaras disponibles
        print(f"   🔄 Buscando cámara alternativa...")
        for camera_id in available_cameras:
            if camera_id == self.requested_id:
                continue  # Ya probada
            
            print(f"   🔍 Probando cámara {camera_id}...")
            success, cap = self._test_camera(camera_id)
            
            if success:
                self.cap = cap
                self.actual_id = camera_id
                print(f"   ✅ Usando cámara {camera_id} como alternativa")
                self._configure_camera()
                return
        
        # 4. Ninguna cámara funcionó
        raise RuntimeError("❌ Ninguna cámara disponible pudo inicializarse correctamente")
    
    def _configure_camera(self):
        """Configura resolución y obtiene valores reales."""
        # Intentar configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Obtener resolución real (puede ser diferente a la solicitada)
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Mostrar información
        print(f"   ✅ Cámara {self.actual_id} inicializada")
        print(f"   📐 Resolución: {self.actual_width}x{self.actual_height}", end="")
        
        if (self.actual_width, self.actual_height) != (self.width, self.height):
            print(f" (solicitado: {self.width}x{self.height})")
        else:
            print()
        
        # Obtener FPS si está disponible
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            print(f"   🎬 FPS: {fps:.1f}")
    
    def read_frame(self):
        """
        Lee un frame de la cámara.
        
        Returns:
            numpy.ndarray: Frame BGR o None si falla
        """
        if not self.cap or not self.cap.isOpened():
            print("⚠️ Cámara no disponible")
            return None
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        
        return cv2.flip(frame, 1)  # Espejo horizontal
    
    def release(self):
        """Libera la cámara."""
        if self.cap:
            self.cap.release()
            print(f"📹 Cámara {self.actual_id} liberada")
    
    def get_info(self):
        """
        Obtiene información de la cámara actual.
        
        Returns:
            dict: Información de la cámara
        """
        return {
            'id': self.actual_id,
            'requested_id': self.requested_id,
            'width': self.actual_width,
            'height': self.actual_height,
            'is_open': self.cap.isOpened() if self.cap else False,
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
