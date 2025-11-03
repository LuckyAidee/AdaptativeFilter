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

