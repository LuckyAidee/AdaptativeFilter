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
        
        def measure_parallel(self, video_source=0):
            print("\n" + "=" * 70)
            print("MIDIENDO RENDIMIENTO: VERSIÓN PARALELA (2 PROCESOS - MULTIPROCESSING)")
            print("=" * 70)
        
        from multiprocessing import Value
        detector = ParallelPipelineDetector(camera_id=video_source) 
        detector.running.value = 1
        
        from multiprocessing import Process
        p1 = Process(target=detector.capture_process, args=(detector.camera_id, detector.frame_queue, detector.running))
        p2 = Process(target=detector.processing_process, args=(detector.frame_queue, detector.result_queue, detector.running))
        
        p1.start()
        p2.start()
        
        frame_count = 0
        frame_times = []
        
        try:
            time.sleep(1.0)
            
            start_time = time.time()
            
            while frame_count < self.num_frames:
                try:
                    result_data = detector.result_queue.get(timeout=1.0)
                    frame_times.append(time.time())
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        print(f"Procesados: {frame_count}/{self.num_frames} frames")
                        
                except:
                    continue
            
            total_time = time.time() - start_time
            
            # Calcular tiempo promedio por cada frame
            if len(frame_times) > 1:
                intervals = np.diff(frame_times)
                avg_interval = np.mean(intervals)
                self.results['parallel']['avg_time'] = avg_interval
                self.results['parallel']['fps'] = 1.0 / avg_interval
            
        finally:
            detector.running.value = 0
            time.sleep(0.5)
            p1.terminate()
            p2.terminate()
            p1.join(timeout=1)
            p2.join(timeout=1)
        
        print(f"\nTiempo promedio: {self.results['parallel']['avg_time']*1000:.2f} ms/frame")
        print(f"FPS promedio: {self.results['parallel']['fps']:.2f}")
        print("=" * 70)