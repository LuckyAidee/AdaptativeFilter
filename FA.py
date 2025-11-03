import cv2
import numpy as np
import time
from multiprocessing import Process, Queue, Value
from collections import deque
from queue import Empty
import os

class AdaptiveFilter:
     # Filtro adaptativo 
    def __init__(self):
        self.history = deque(maxlen=3)
        self.times = []
        self.active_filter = "none"

        # Inicializa los kernels principales
        self._init_kernels()

        # Resolución reducida para análisis más rápido
        self.resolution = (320, 240)
      # Preparar los kernels base usados en los filtros.
    def _init_kernels(self):
        # Kernel de detección de bordes
        self.edge_kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

        # Kernel gaussiano
        self.gaussian_kernel = cv2.getGaussianKernel(3, 0.5)

        print("Kernels listos")

     # Analiza el fotograma y estima condiciones de luz, niebla y lluvia.
    def analyze_frame(self, frame):
        resized = cv2.resize(frame, self.resolution)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        brightness = np.mean(gray)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Luz baja 
        low_light = brightness < 80

        # Niebla 
        edges = cv2.Canny(gray, 40, 120)
        edge_density = np.sum(edges) / edges.size
        fog = edge_density < 0.05 and brightness > 60

        # Lluvia 
        gray_blur = cv2.medianBlur(gray, 5)
        rain_edges = cv2.Canny(gray_blur, 40, 120)
        lines = cv2.HoughLinesP(rain_edges, 1, np.pi / 180, 15, minLineLength=20, maxLineGap=10)
        rain = lines is not None and len(lines) > 10

        return {
            "low_light": low_light,
            "fog": fog,
            "rain": rain,
            "brightness": brightness,
            "blur": blur_value
        }
    
    # Promedia las últimas detecciones para evitar cambios bruscos
    def _smooth_conditions(self, conditions):
        
        self.history.append(conditions)
        if len(self.history) < 2:
            return conditions

        smoothed = {}
        for key in ["low_light", "fog", "rain"]:
            votes = sum(1 for c in self.history if c[key])
            smoothed[key] = votes >= len(self.history) // 2

        smoothed.update({
            "brightness": conditions["brightness"],
            "blur": conditions["blur"]
        })
        return smoothed
    
    #LIME: Mejora zonas oscuras en condiciones de poca luz.
    def _apply_lime(self, frame):

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * 0.5), int(h * 0.5)))
        img = small.astype(np.float32) / 255.0

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        illumination = cv2.bilateralFilter(gray, 5, 100, 100)
        illumination = np.clip(illumination.astype(np.float32) / 255.0, 0.2, 1.0)

        enhanced = np.zeros_like(img)
        for i in range(3):
            enhanced[:, :, i] = np.power(img[:, :, i] / illumination, 0.6)

        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        return cv2.resize(enhanced, (w, h))
    
    #FVR: Reducir la neblina
    def _apply_fvr(self, frame):

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * 0.6), int(h * 0.6)))
        img = small.astype(np.float64) / 255.0

        airlight = np.percentile(small, 95, axis=(0, 1))
        dark_channel = np.min(img, axis=2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dark_channel = cv2.morphologyEx(dark_channel, cv2.MORPH_ERODE, kernel)

        transmission = 1 - 0.95 * dark_channel / (airlight.max() / 255.0)
        transmission = np.maximum(transmission, 0.2)

        restored = np.zeros_like(img)
        for i in range(3):
            restored[:, :, i] = (img[:, :, i] - airlight[i] / 255.0) / transmission + airlight[i] / 255.0

        result = np.clip(restored * 255, 0, 255).astype(np.uint8)
        return cv2.resize(result, (w, h))
    
    #ARR: Elimina lluvia mediante filtrado morfológico e inpainting
    def _apply_arr(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        rain_mask = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_v)

        _, rain_mask = cv2.threshold(rain_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        rain_mask = cv2.dilate(rain_mask, kernel_dilate, iterations=1)

        clean = cv2.inpaint(frame, rain_mask, 3, cv2.INPAINT_TELEA)
        smooth = cv2.bilateralFilter(clean, 5, 50, 50)
        return smooth
    
    # Seleccionar y aplicar el filtro adecuado según las condiciones
    def apply_filter(self, frame, conditions):

        start = time.time()

        if conditions["rain"]:
            output = self._apply_arr(frame)
            self.active_filter = "ARR (Eliminación de lluvia)"
        elif conditions["fog"]:
            output = self._apply_fvr(frame)
            self.active_filter = "FVR (Corrección de neblina)"
        elif conditions["low_light"]:
            output = self._apply_lime(frame)
            self.active_filter = "LIME (Mejora de baja luz)"
        else:
            output = frame.copy()
            self.active_filter = "Ninguno"

        self.times.append(time.time() - start)
        return output
    

