import cv2
import numpy as np
import time
from multiprocessing import Process, Queue, Value, shared_memory
from collections import deque
from queue import Empty
import os

class OptimizedAdaptiveFilter:
    """
    Filtro Adaptativo OPTIMIZADO con resolución reducida y algoritmos simplificados
    """

    def __init__(self):
        self.condition_history = deque(maxlen=3)
        self.processing_times = []
        self.current_filter = "none"

        # OPTIMIZACIÓN: Pre-computar kernels
        self.setup_kernels()

        # OPTIMIZACIÓN: Resolución de análisis reducida
        self.analysis_resolution = (320, 240)

    def setup_kernels(self):
        """Pre-computa kernels optimizados (más pequeños)"""
        self.edge_kernel = np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]], dtype=np.float32)

        self.gaussian_kernel = cv2.getGaussianKernel(3, 0.5)

        print("Filtros optimizados inicializados")

    def analyze_conditions(self, frame):
        """Análisis OPTIMIZADO con resolución reducida"""
        start_time = time.time()

        # OPTIMIZACIÓN 1: Reducir resolución para análisis
        small_frame = cv2.resize(frame, self.analysis_resolution)

        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        h, w = gray.shape

        center_region = gray[h//4:3*h//4, w//4:3*w//4]

        # Métricas de iluminación
        global_brightness = np.mean(gray)
        center_brightness = np.mean(center_region)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / hist.sum()

        dark_pixels = np.sum(hist_norm[:64])
        bright_pixels = np.sum(hist_norm[192:])
        noise_level = np.std(center_region)

        # Detección de baja iluminación
        low_light_criteria = [
            global_brightness < 60,
            center_brightness < 50,
            dark_pixels > 0.80,
            bright_pixels < 0.03,
            noise_level > 25
        ]
        low_light_confidence = sum(low_light_criteria) / len(low_light_criteria)

        if global_brightness < 60 and dark_pixels > 0.7:
            low_light_confidence = min(low_light_confidence + 0.3, 1.0)

        # Detección de niebla
        laplacian_var = cv2.Laplacian(center_region, cv2.CV_64F).var()

        fog_confidence = 0.0
        if global_brightness > 70:
            sobelx = cv2.Sobel(center_region, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(center_region, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mean = np.mean(np.abs(sobelx) + np.abs(sobely))

            local_std = np.std(center_region)
            avg_saturation = np.mean(hsv[:, :, 1])

            fog_criteria = [
                laplacian_var < 500,
                sobel_mean < 20,
                local_std < 35,
                avg_saturation < 120,
                global_brightness > 50
            ]
            fog_confidence = sum(fog_criteria) / len(fog_criteria)

        # Detección de lluvia
        rain_confidence = 0.0
        rain_features = {'vertical_lines': 0, 'short_segments': 0, 'density_score': 0}

        if global_brightness > 60 and noise_level < 20 and low_light_confidence < 0.4:
            filtered_gray = cv2.medianBlur(gray, 3)
            edges = cv2.Canny(filtered_gray, 40, 120)

            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                                   minLineLength=20, maxLineGap=5)

            if lines is not None:
                vertical_lines = 0
                diagonal_lines = 0

                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)

                    if 85 <= angle <= 95 and length > 20:
                        vertical_lines += 1
                    elif 70 <= angle <= 110 and length > 25:
                        diagonal_lines += 1

                rain_features['vertical_lines'] = vertical_lines
                rain_features['diagonal_lines'] = diagonal_lines

                if len(lines) > 0:
                    relevant_lines = vertical_lines + diagonal_lines
                    rain_features['density_score'] = relevant_lines / len(lines)

            rain_criteria = [
                rain_features['vertical_lines'] >= 12,
                rain_features['density_score'] > 0.6,
                global_brightness > 60,
                noise_level < 15,
                low_light_confidence < 0.3
            ]

            if all(rain_criteria):
                rain_confidence = 0.9

        # Decisión final
        final_decisions = self.apply_decision_logic(
            low_light_confidence, fog_confidence, rain_confidence,
            global_brightness, laplacian_var, rain_features
        )

        conditions = {
            'low_light': final_decisions['low_light'],
            'fog': final_decisions['fog'],
            'rain': final_decisions['rain'],
            'brightness_value': global_brightness,
            'blur_value': laplacian_var,
            'rain_score': rain_confidence,
            'low_light_confidence': low_light_confidence,
            'fog_confidence': fog_confidence,
            'rain_confidence': rain_confidence,
            'processing_time': time.time() - start_time
        }

        return conditions

    def apply_decision_logic(self, low_light_conf, fog_conf, rain_conf, brightness, blur_val, rain_features):
        """Lógica de decisión con prioridad LIME"""
        decisions = {'low_light': False, 'fog': False, 'rain': False, 'combination': 'none'}

        if brightness < 85 or low_light_conf >= 0.6:
            decisions['low_light'] = True
            decisions['combination'] = 'light_priority'
            return decisions

        if (rain_conf >= 0.85 and brightness >= 70 and low_light_conf < 0.3 and
            rain_features.get('vertical_lines', 0) >= 12):
            decisions['rain'] = True
            decisions['combination'] = 'rain_confirmed'
            return decisions

        if (fog_conf >= 0.7 and brightness >= 75 and low_light_conf < 0.2 and rain_conf < 0.3):
            decisions['fog'] = True
            decisions['combination'] = 'fog_clear'
            return decisions

        decisions['combination'] = 'normal_conditions'
        return decisions

    def temporal_smoothing(self, conditions):
        """Suavizado temporal para evitar cambios abruptos"""
        self.condition_history.append(conditions)

        if len(self.condition_history) < 2:
            return conditions

        avg_conditions = {}
        for key in ['low_light', 'fog', 'rain']:
            votes = sum(1 for c in self.condition_history if c[key])
            avg_conditions[key] = votes >= len(self.condition_history) // 2

        avg_conditions.update({
            'brightness_value': conditions['brightness_value'],
            'blur_value': conditions['blur_value'],
            'rain_score': conditions['rain_score'],
            'low_light_confidence': conditions['low_light_confidence'],
            'fog_confidence': conditions['fog_confidence'],
            'rain_confidence': conditions['rain_confidence']
        })

        return avg_conditions

    def apply_LIME_optimized(self, frame):
        """LIME OPTIMIZADO - Procesa en resolución menor y escala"""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w//2, h//2))

        img_float = small.astype(np.float64) / 255.0

        gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

        illumination = cv2.bilateralFilter(gray, 5, 80, 80)
        illumination = illumination.astype(np.float64) / 255.0
        illumination = np.maximum(illumination, 0.2)

        enhanced = np.zeros_like(img_float)
        for i in range(3):
            channel = img_float[:, :, i]
            enhanced[:, :, i] = np.power(channel / illumination[:, :, np.newaxis].squeeze(), 0.2)

        enhanced = np.power(enhanced, 1.2)
        enhanced = np.clip(enhanced, 0, 0.75)

        blend_ratio = 0.3
        enhanced = enhanced * blend_ratio + img_float * (1 - blend_ratio)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        enhanced = cv2.resize(enhanced, (w, h))

        return enhanced

    def apply_FVR_optimized(self, frame):
        """FVR OPTIMIZADO - Eliminación rápida de niebla"""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w*0.6), int(h*0.6)))

        img_float = small.astype(np.float64) / 255.0

        airlight = np.percentile(small, 95, axis=(0, 1))
        dark_channel = np.min(img_float, axis=2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dark_channel = cv2.morphologyEx(dark_channel, cv2.MORPH_ERODE, kernel)

        transmission = 1 - 0.95 * dark_channel / (airlight.max() / 255.0)
        transmission = np.maximum(transmission, 0.2)

        result = np.zeros_like(img_float)
        for i in range(3):
            result[:, :, i] = (img_float[:, :, i] - airlight[i]/255.0) / transmission + airlight[i]/255.0

        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        result = cv2.resize(result, (w, h))

        return result

    def apply_ARR_optimized(self, frame):
        """ARR OPTIMIZADO - Eliminación rápida de lluvia"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        rain_mask = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_vertical)

        _, rain_mask = cv2.threshold(rain_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        rain_mask = cv2.dilate(rain_mask, kernel_dilate, iterations=1)

        result = cv2.inpaint(frame, rain_mask, 3, cv2.INPAINT_TELEA)
        result = cv2.bilateralFilter(result, 5, 50, 50)

        return result

    def apply_filter(self, frame, conditions):
        """Aplicar filtro optimizado según condiciones"""
        start_time = time.time()

        if conditions['rain']:
            filtered_frame = self.apply_ARR_optimized(frame)
            self.current_filter = "ARR (Rain Removal)"
        elif conditions['fog']:
            filtered_frame = self.apply_FVR_optimized(frame)
            self.current_filter = "FVR (Fog Removal)"
        elif conditions['low_light']:
            filtered_frame = self.apply_LIME_optimized(frame)
            self.current_filter = "LIME (Low Light)"
        else:
            filtered_frame = frame.copy()
            self.current_filter = "Sin filtro"

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]

        return filtered_frame


class ParallelPipelineDetector:
    """
    Sistema con PIPELINE DE 2 PROCESOS con optimizaciones:
    - Proceso 1: Captura de frames
    - Proceso 2: Procesamiento completo (Análisis + Filtrado + Suavizado)

    OPTIMIZACIONES APLICADAS:
    1. Queue size aumentado de 2 → 10
    2. Timeouts aumentados de 0.01 → 0.05
    3. Shared memory para evitar copiar frames
    """

    def __init__(self, camera_id=0, use_shared_memory=True):
        self.camera_id = camera_id
        self.cap = None
        self.adaptive_filter = OptimizedAdaptiveFilter()
        self.use_shared_memory = use_shared_memory

        # OPTIMIZACIÓN 1: Aumentar queue size para mejor pipelining
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)

        self.running = Value('i', 0)

        # OPTIMIZACIÓN 3: Shared Memory para frames
        self.shm_buffers = []
        self.shm_index = Value('i', 0)
        if use_shared_memory:
            self._setup_shared_memory()

        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        self.processes = []

    def _setup_shared_memory(self):
        """Configura buffers de shared memory para frames (640x480x3)"""
        try:
            frame_size = 640 * 480 * 3
            self.shm_buffers = []

            for i in range(10):
                try:
                    shm = shared_memory.SharedMemory(create=True, size=frame_size)
                    self.shm_buffers.append({
                        'shm': shm,
                        'name': shm.name,
                        'array': np.ndarray((480, 640, 3), dtype=np.uint8, buffer=shm.buf)
                    })
                except Exception as e:
                    print(f"Warning: No se pudo crear shared memory buffer {i}: {e}")
                    self.use_shared_memory = False
                    self._cleanup_shared_memory()
                    break

            if self.use_shared_memory:
                print(f"✓ Shared memory habilitada: {len(self.shm_buffers)} buffers de {frame_size/1024:.1f}KB")
        except Exception as e:
            print(f"Warning: Shared memory no disponible: {e}. Usando Queue estándar.")
            self.use_shared_memory = False

    def _cleanup_shared_memory(self):
        """Limpia buffers de shared memory"""
        for buf in self.shm_buffers:
            try:
                buf['shm'].close()
                buf['shm'].unlink()
            except:
                pass
        self.shm_buffers = []

    def initialize_camera(self):
        """Inicializar cámara optimizada"""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise Exception(f"No se pudo abrir la cámara {self.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"Cámara: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    @staticmethod
    def capture_process(camera_id, frame_queue, running, shm_buffers_info=None, shm_index=None):
        """Proceso de captura de frames (con soporte para shared memory)"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        use_shm = shm_buffers_info is not None and shm_index is not None
        shm_arrays = []

        if use_shm:
            try:
                for buf_info in shm_buffers_info:
                    shm = shared_memory.SharedMemory(name=buf_info['name'])
                    arr = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=shm.buf)
                    shm_arrays.append({'shm': shm, 'array': arr})
            except Exception as e:
                print(f"Warning: Shared memory no disponible en capture_process: {e}")
                use_shm = False

        while running.value == 1:
            ret, frame = cap.read()
            if ret:
                try:
                    if use_shm:
                        # OPTIMIZACIÓN 3: Escribir directamente en shared memory
                        idx = shm_index.value % len(shm_arrays)
                        np.copyto(shm_arrays[idx]['array'], frame)
                        # OPTIMIZACIÓN 2: Timeout aumentado
                        frame_queue.put(idx, timeout=0.05)
                        shm_index.value += 1
                    else:
                        # OPTIMIZACIÓN 2: Timeout aumentado
                        frame_queue.put(frame, timeout=0.05)
                except Exception:
                    pass
            else:
                time.sleep(0.01)

        cap.release()
        if use_shm:
            for shm_arr in shm_arrays:
                try:
                    shm_arr['shm'].close()
                except:
                    pass

    @staticmethod
    def processing_process(frame_queue, result_queue, running, shm_buffers_info=None):
        """Proceso de procesamiento completo (con soporte para shared memory)"""
        adaptive_filter = OptimizedAdaptiveFilter()

        use_shm = shm_buffers_info is not None
        shm_arrays = []

        if use_shm:
            try:
                for buf_info in shm_buffers_info:
                    shm = shared_memory.SharedMemory(name=buf_info['name'])
                    arr = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=shm.buf)
                    shm_arrays.append({'shm': shm, 'array': arr})
            except Exception as e:
                print(f"Warning: Shared memory no disponible en processing_process: {e}")
                use_shm = False

        while running.value == 1:
            try:
                data = frame_queue.get(timeout=0.1)

                if use_shm and isinstance(data, int):
                    # OPTIMIZACIÓN 3: Leer directamente desde shared memory
                    idx = data % len(shm_arrays)
                    frame = shm_arrays[idx]['array'].copy()
                else:
                    frame = data

                conditions = adaptive_filter.analyze_conditions(frame)
                smooth_conditions = adaptive_filter.temporal_smoothing(conditions)
                filtered_frame = adaptive_filter.apply_filter(frame, smooth_conditions)

                result_data = {
                    'original': frame,
                    'filtered': filtered_frame,
                    'conditions': smooth_conditions,
                    'filter_used': adaptive_filter.current_filter
                }

                try:
                    # OPTIMIZACIÓN 2: Timeout aumentado
                    result_queue.put(result_data, timeout=0.05)
                except Exception:
                    pass

            except Empty:
                continue
            except KeyboardInterrupt:
                break
            except Exception:
                continue

        if use_shm:
            for shm_arr in shm_arrays:
                try:
                    shm_arr['shm'].close()
                except:
                    pass

    def add_info_overlay(self, frame, conditions, filter_used):
        """Interfaz minimalista con información"""
        overlay_frame = frame.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, overlay_frame, 0.4, 0, overlay_frame)

        y_offset = 25
        cv2.putText(overlay_frame, f"FPS: {self.current_fps:.0f} | {filter_used}",
                   (15, y_offset), font, font_scale, (255, 255, 255), thickness)

        y_offset += 15
        brightness_color = (255, 100, 100) if conditions['brightness_value'] < 85 else (255, 255, 255)
        cv2.putText(overlay_frame, f"Brillo: {conditions['brightness_value']:.0f}",
                   (15, y_offset), font, font_scale, brightness_color, thickness)

        status_colors = {
            'low_light': (100, 255, 255) if conditions['low_light'] else (60, 60, 60),
            'fog': (255, 100, 255) if conditions['fog'] else (60, 60, 60),
            'rain': (100, 255, 100) if conditions['rain'] else (60, 60, 60)
        }

        cv2.circle(overlay_frame, (15, 60), 6, status_colors['low_light'], -1)
        cv2.putText(overlay_frame, "L", (12, 63), font, 0.3, (0, 0, 0), 1)

        cv2.circle(overlay_frame, (40, 60), 6, status_colors['fog'], -1)
        cv2.putText(overlay_frame, "F", (37, 63), font, 0.3, (0, 0, 0), 1)

        cv2.circle(overlay_frame, (65, 60), 6, status_colors['rain'], -1)
        cv2.putText(overlay_frame, "R", (62, 63), font, 0.3, (0, 0, 0), 1)

        return overlay_frame

    def calculate_fps(self):
        """Calcular FPS"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time

    def run(self):
        """Ejecutar sistema con pipeline de 2 procesos (con shared memory opcional)"""
        try:
            self.running.value = 1

            if self.use_shared_memory and self.shm_buffers:
                shm_info = [{'name': buf['name']} for buf in self.shm_buffers]
                p1 = Process(target=self.capture_process,
                           args=(self.camera_id, self.frame_queue, self.running, shm_info, self.shm_index))
                p2 = Process(target=self.processing_process,
                           args=(self.frame_queue, self.result_queue, self.running, shm_info))
                mode_desc = "con shared memory"
            else:
                p1 = Process(target=self.capture_process,
                           args=(self.camera_id, self.frame_queue, self.running))
                p2 = Process(target=self.processing_process,
                           args=(self.frame_queue, self.result_queue, self.running))
                mode_desc = "con Queue"

            p1.start()
            p2.start()

            self.processes = [p1, p2]

            print(f"Sistema PIPELINE 2 PROCESOS iniciado (multiprocessing {mode_desc})")
            print("Presiona 'q' para salir, 's' para guardar")

            while self.running.value == 1:
                try:
                    result_data = self.result_queue.get(timeout=0.1)

                    original = result_data['original']
                    filtered = result_data['filtered']
                    conditions = result_data['conditions']
                    filter_used = result_data['filter_used']

                    display_frame = self.add_info_overlay(filtered, conditions, filter_used)
                    combined = np.hstack([original, display_frame])

                    if combined.shape[1] > 1200:
                        scale = 1200 / combined.shape[1]
                        new_width = int(combined.shape[1] * scale)
                        new_height = int(combined.shape[0] * scale)
                        combined = cv2.resize(combined, (new_width, new_height))

                    cv2.imshow('Filtro Adaptativo - Original vs Filtrado', combined)

                    self.calculate_fps()

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = int(time.time())
                        cv2.imwrite(f'resultado_{timestamp}.jpg', combined)
                        print(f"Frame guardado: resultado_{timestamp}.jpg")

                except KeyboardInterrupt:
                    break
                except Empty:
                    continue
                except Exception:
                    continue

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Limpiar recursos"""
        self.running.value = 0
        time.sleep(0.5)

        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

        if self.use_shared_memory:
            self._cleanup_shared_memory()

        cv2.destroyAllWindows()
        print("\nSistema detenido")


def main():
    """
    Sistema Optimizado con Pipeline de 2 Procesos (Multiprocessing)
    """
    print("=" * 80)
    print("FILTRO ADAPTATIVO PARALELO - MULTIPROCESSING (2 PROCESOS)")
    print("=" * 80)
    print("ARQUITECTURA: Pipeline de 2 procesos independientes")
    print("OPTIMIZACIONES:")
    print("  - Análisis en resolución reducida (320x240)")
    print("  - Algoritmos LIME/FVR/ARR simplificados")
    print("  - Queue size aumentado (2 → 10)")
    print("  - Timeouts optimizados (0.01 → 0.05)")
    print("  - Shared memory para frames")
    print("=" * 80)
    print("PROCESOS DEL PIPELINE:")
    print("  1. Captura de Frames (I/O)")
    print("  2. Procesamiento Completo (Análisis + Filtrado + Suavizado)")
    print("=" * 80)

    try:
        detector = ParallelPipelineDetector(camera_id=0)
        detector.run()

    except KeyboardInterrupt:
        print("\nDetenido por el usuario")
    except Exception as e:
        print(f"Error fatal: {e}")

if __name__ == "__main__":
    main()
