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
        
def calculate_metrics(self):
        seq_time = self.results['sequential']['avg_time']
        par_time = self.results['parallel']['avg_time']
        
        if par_time > 0:
            self.results['speedup'] = seq_time / par_time
            
            self.results['efficiency'] = self.results['speedup'] / self.results['num_threads']
        
        if seq_time > 0:
            latency_reduction = ((seq_time - par_time) / seq_time) * 100
            self.results['latency_reduction'] = latency_reduction
        else:
            self.results['latency_reduction'] = 0
    
def print_results(self):

        print("\n" + "=" * 70)
        print("RESULTADOS DE RENDIMIENTO")
        print("=" * 70)
        
        print("\nVersión Secuencial:")
        print(f"  Tiempo promedio: {self.results['sequential']['avg_time']*1000:.2f} ms/frame")
        print(f"  FPS:             {self.results['sequential']['fps']:.2f}")
        
        print("\nVersión Paralela (2 procesos - multiprocessing):")
        print(f"  Tiempo promedio: {self.results['parallel']['avg_time']*1000:.2f} ms/frame")
        print(f"  FPS:             {self.results['parallel']['fps']:.2f}")
        
        print("\n" + "-" * 70)
        print("MÉTRICAS DE PARALELIZACIÓN:")
        print("-" * 70)
        print(f"Speedup:            {self.results['speedup']:.2f}x")
        print(f"Eficiencia:         {self.results['efficiency']*100:.1f}%")
        print(f"Reducción latencia: {self.results['latency_reduction']:.1f}%")
        print(f"Número de hilos:    {self.results['num_threads']}")
        
        print("\n" + "=" * 70)
        
        print("\nEVALUACIÓN VS OBJETIVOS DEL PROTOCOLO:")
        print("-" * 70)
        
        if self.results['latency_reduction'] > 40:
            print(f"✓ Reducción latencia: {self.results['latency_reduction']:.1f}% (Objetivo: >40%)")
        else:
            print(f"✗ Reducción latencia: {self.results['latency_reduction']:.1f}% (Objetivo: >40%)")
        
        if self.results['parallel']['fps'] >= 15:
            print(f"✓ Throughput: {self.results['parallel']['fps']:.1f} FPS (Objetivo: ≥15 FPS)")
        else:
            print(f"✗ Throughput: {self.results['parallel']['fps']:.1f} FPS (Objetivo: ≥15 FPS)")
        
        print("=" * 70)

def plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Análisis de Rendimiento: Secuencial vs Paralelo', fontsize=16)
        
        ax1 = axes[0, 0]
        categories = ['Secuencial', 'Paralelo']
        times_ms = [
            self.results['sequential']['avg_time'] * 1000,
            self.results['parallel']['avg_time'] * 1000
        ]
        colors = ['#FF6B6B', '#4ECDC4']
        bars = ax1.bar(categories, times_ms, color=colors, alpha=0.7)
        ax1.set_ylabel('Tiempo (ms/frame)')
        ax1.set_title('Tiempo de Procesamiento por Frame')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}ms', ha='center', va='bottom')
        
        ax2 = axes[0, 1]
        fps_values = [
            self.results['sequential']['fps'],
            self.results['parallel']['fps']
        ]
        bars = ax2.bar(categories, fps_values, color=colors, alpha=0.7)
        ax2.set_ylabel('Frames por Segundo')
        ax2.set_title('Throughput (FPS)')
        ax2.axhline(y=15, color='green', linestyle='--', label='Objetivo: 15 FPS')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend()
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}', ha='center', va='bottom')
        
        ax3 = axes[1, 0]
        speedup = self.results['speedup']
        ideal_speedup = self.results['num_threads']
        
        x = ['Speedup\nObtenido', 'Speedup\nIdeal']
        y = [speedup, ideal_speedup]
        colors_speedup = ['#4ECDC4', '#FFE66D']
        bars = ax3.bar(x, y, color=colors_speedup, alpha=0.7)
        ax3.set_ylabel('Speedup')
        ax3.set_title('Speedup vs Ideal')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}x', ha='center', va='bottom')
        
        ax4 = axes[1, 1]
        efficiency = self.results['efficiency'] * 100
        ideal_efficiency = 100
        
        x = ['Eficiencia\nObtenida', 'Eficiencia\nIdeal']
        y = [efficiency, ideal_efficiency]
        colors_eff = ['#4ECDC4', '#FFE66D']
        bars = ax4.bar(x, y, color=colors_eff, alpha=0.7)
        ax4.set_ylabel('Eficiencia (%)')
        ax4.set_title('Eficiencia del Paralelismo')
        ax4.set_ylim([0, 110])
        ax4.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('performance_results.png', dpi=300, bbox_inches='tight')
        print("\nGráficas guardadas en: performance_results.png")
        plt.show()