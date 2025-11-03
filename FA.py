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

        print("Kernels listos.")
