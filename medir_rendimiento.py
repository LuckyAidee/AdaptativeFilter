import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from FA import OptimizedAdaptiveFilter, ParallelPipelineDetector
from baseline_secuencial import SequentialBaseline
import json
from multiprocessing import Process, Queue, Value 
from queue import Empty 
import os

class PerformanceAnalyzer:
    def __init__(self, num_frames=100):
        self.num_frames = num_frames
        self.results = {
            'sequential': {
                'times': [],
                'avg_time': 0,
                'fps': 0
            },
            'parallel': {
                'times': [],
                'avg_time': 0,
                'fps': 0
            },
            'speedup': 0,
            'efficiency': 0,
            'num_threads': 2  #Capturar y procesar
        }
        
        def measure_sequential(self, video_source=0):
            print("\n" + "=" * 70)
            print("MIDIENDO RENDIMIENTO: VERSIÓN SECUENCIAL")
            print("=" * 70)
        
        baseline = SequentialBaseline(camera_id=video_source) 
        baseline.initialize_camera()
        
        frame_count = 0
        
        try:
            while frame_count < self.num_frames:
                ret, frame = baseline.cap.read()
                if not ret:
                    continue
                
                _, _, proc_time = baseline.process_frame_sequential(frame)
                
                self.results['sequential']['times'].append(proc_time)
                frame_count += 1
                
                if frame_count % 10 == 0:
                    print(f"Procesados: {frame_count}/{self.num_frames} frames")
        
        finally:
            baseline.cleanup()
        
        #Calcular estadísticas
        times = self.results['sequential']['times']
        if times:
            self.results['sequential']['avg_time'] = np.mean(times)
            self.results['sequential']['fps'] = 1.0 / np.mean(times)
        
        print(f"\nTiempo promedio: {self.results['sequential']['avg_time']*1000:.2f} ms/frame")
        print(f"FPS promedio: {self.results['sequential']['fps']:.2f}")
        print("=" * 70)