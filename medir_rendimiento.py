import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from FA import OptimizedAdaptiveFilter, ParallelPipelineDetector
from baseline_secuencial import SequentialBaseline
import json
import argparse
from glob import glob
from pathlib import Path
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

    def measure_sequential(self, video_source=0, image_folder=None):
        """
        Medir rendimiento de versión secuencial
        Args:
            video_source: ID de cámara (default: 0)
            image_folder: Carpeta con imágenes para testing (None = usar cámara)
        """
        print("\n" + "=" * 70)
        print("MIDIENDO RENDIMIENTO: VERSIÓN SECUENCIAL")
        print("=" * 70)

        # Modo dataset: procesar imágenes desde carpeta
        if image_folder:
            print(f"Modo: Dataset desde {image_folder}")

            # Buscar todas las imágenes
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_paths.extend(glob(f"{image_folder}/**/{ext}", recursive=True))

            if not image_paths:
                print(f"✗ No se encontraron imágenes en {image_folder}")
                return

            # Limitar al número de frames solicitado
            image_paths = image_paths[:self.num_frames]
            print(f"Total imágenes a procesar: {len(image_paths)}")

            baseline = SequentialBaseline(camera_id=0)

            start_time = time.time()

            for i, img_path in enumerate(image_paths):
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                _, _, proc_time = baseline.process_frame_sequential(frame)
                self.results['sequential']['times'].append(proc_time)

                if (i + 1) % 100 == 0:
                    print(f"Procesados: {i + 1}/{len(image_paths)} frames")

            total_time = time.time() - start_time
            print(f"\nProcesados: {len(image_paths)} frames en {total_time:.2f}s")

        # Modo cámara: comportamiento original
        else:
            print("Modo: Cámara en vivo")
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

        # Calcular estadísticas
        times = self.results['sequential']['times']
        if times:
            self.results['sequential']['avg_time'] = np.mean(times)
            self.results['sequential']['fps'] = 1.0 / np.mean(times)

        print(f"\nTiempo promedio: {self.results['sequential']['avg_time']*1000:.2f} ms/frame")
        print(f"FPS promedio: {self.results['sequential']['fps']:.2f}")
        print("=" * 70)

    def measure_parallel(self, video_source=0, image_folder=None):
        """
        Medir rendimiento de versión paralela con multiprocessing
        Args:
            video_source: ID de cámara (default: 0)
            image_folder: Carpeta con imágenes para testing (None = usar cámara)
        """
        print("\n" + "=" * 70)
        print("MIDIENDO RENDIMIENTO: VERSIÓN PARALELA (2 PROCESOS - MULTIPROCESSING)")
        print("=" * 70)

        # Modo dataset: alimentar imágenes al pipeline
        if image_folder:
            print(f"Modo: Dataset desde {image_folder}")

            # Buscar todas las imágenes
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_paths.extend(glob(f"{image_folder}/**/{ext}", recursive=True))

            if not image_paths:
                print(f"✗ No se encontraron imágenes en {image_folder}")
                return

            # Limitar al número de frames solicitado
            image_paths = image_paths[:self.num_frames]
            print(f"Total imágenes a procesar: {len(image_paths)}")

            detector = ParallelPipelineDetector(camera_id=0)
            detector.running.value = 1

            # Solo iniciar proceso de PROCESAMIENTO (no captura)
            p2 = Process(target=detector.processing_process,
                        args=(detector.frame_queue, detector.result_queue, detector.running))
            p2.start()

            frame_count = 0
            frame_times = []

            try:
                time.sleep(0.5)  # Esperar que el proceso inicie

                start_time = time.time()

                # Alimentar imágenes al queue desde el proceso principal
                images_sent = 0
                for img_path in image_paths:
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue

                    # Enviar frame al queue
                    try:
                        detector.frame_queue.put(frame, timeout=0.1)
                        images_sent += 1
                    except:
                        pass  # Queue lleno, continuar

                    # Recoger resultados procesados
                    try:
                        result_data = detector.result_queue.get(timeout=0.01)
                        frame_times.append(time.time())
                        frame_count += 1

                        if frame_count % 100 == 0:
                            print(f"Procesados: {frame_count}/{len(image_paths)} frames")
                    except:
                        pass  # No hay resultados todavía

                # Recoger resultados restantes
                while frame_count < images_sent:
                    try:
                        result_data = detector.result_queue.get(timeout=1.0)
                        frame_times.append(time.time())
                        frame_count += 1

                        if frame_count % 100 == 0:
                            print(f"Procesados: {frame_count}/{len(image_paths)} frames")
                    except:
                        break

                total_time = time.time() - start_time

                # Calcular tiempo promedio por frame
                if len(frame_times) > 1:
                    intervals = np.diff(frame_times)
                    avg_interval = np.mean(intervals)
                    self.results['parallel']['avg_time'] = avg_interval
                    self.results['parallel']['fps'] = 1.0 / avg_interval

            finally:
                detector.running.value = 0
                time.sleep(0.5)
                p2.terminate()
                p2.join(timeout=1)

        # Modo cámara: comportamiento original
        else:
            print("Modo: Cámara en vivo")
            detector = ParallelPipelineDetector(camera_id=video_source)
            detector.running.value = 1

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

    def save_results(self, filename='performance_results.json'):
        save_data = {
            'sequential': {
                'avg_time_ms': self.results['sequential']['avg_time'] * 1000,
                'fps': self.results['sequential']['fps']
            },
            'parallel': {
                'avg_time_ms': self.results['parallel']['avg_time'] * 1000,
                'fps': self.results['parallel']['fps']
            },
            'speedup': self.results['speedup'],
            'efficiency': self.results['efficiency'] * 100,
            'latency_reduction': self.results['latency_reduction'],
            'num_threads': self.results['num_threads'],
            'num_frames_tested': self.num_frames
        }

        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=4)

        print(f"\nResultados guardados en: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Análisis de rendimiento del filtro adaptativo')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Carpeta con dataset de imágenes (si no se especifica, usa cámara)')
    parser.add_argument('--num-frames', type=int, default=100,
                        help='Número de frames/imágenes a procesar (default: 100)')

    args = parser.parse_args()

    print("=" * 70)
    print("ANÁLISIS DE RENDIMIENTO - MULTIPROCESSING")
    print("=" * 70)
    print("\nEste script medirá:")
    print("  1. Tiempo de procesamiento (Secuencial - sin optimizar)")
    print("  2. Tiempo de procesamiento (Paralelo - 2 procesos)")
    print("  3. Speedup = T_secuencial / T_paralelo")
    print("  4. Eficiencia = Speedup / Núm_procesos")

    if args.dataset:
        print(f"\nModo: Dataset desde {args.dataset}")
        print(f"Número de imágenes a procesar: {args.num_frames}")
    else:
        print(f"\nModo: Cámara en vivo")
        print(f"Número de frames a procesar: {args.num_frames}")

    print("=" * 70)

    if not args.dataset:
        input("\nPresiona ENTER para comenzar la medición...")

    analyzer = PerformanceAnalyzer(num_frames=args.num_frames)

    analyzer.measure_sequential(video_source=0, image_folder=args.dataset)

    print("\nEsperando 3 segundos antes de la siguiente medición...")
    time.sleep(3)

    analyzer.measure_parallel(video_source=0, image_folder=args.dataset)

    analyzer.calculate_metrics()

    analyzer.print_results()

    analyzer.save_results()

    print("\nGenerando gráficas...")
    analyzer.plot_results()

    print("\n¡Análisis completo!")


if __name__ == "__main__":
    main()
