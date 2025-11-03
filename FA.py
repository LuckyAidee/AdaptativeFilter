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
        start_time = time.time()

        # Hacemos un preprocesamiento para reducir la resolución y hacemos una conversión de color

        # Usamos una versión más pequeña del frame para procesar más rápido.
        resized = cv2.resize(frame, self.resolution)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Tomamos solo la región central (Aqui es donde hay más información útil)
        h, w = gray.shape
        center = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]

        #Realizamos un calculo de de luz y ruido
        brightness = float(np.mean(gray))           # Brillo general
        center_brightness = float(np.mean(center))  # Brillo del centro del frame
        noise_level = float(np.std(center))         # Variación de intensidad (ruido)
    
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

class VideoPipeline:

    # Dirige la captura y procesamiento en 2 procesos.
    def __init__(self, camera_id=0):

        self.camera_id = camera_id
        self.frame_queue = Queue(maxsize=8)
        self.result_queue = Queue(maxsize=8)
        self.running = Value('i', 0)
        self.processes = []

        # FPS/UI
        self.fps_counter = 0
        self.current_fps = 0.0
        self.fps_start_time = time.time()
    
    # Procesos 
    @staticmethod
    def capture_process(camera_id, frame_queue, running):
        cap = cv2.VideoCapture(camera_id)
        try:
            while running.value == 1:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                try:
                    frame_queue.put(frame, timeout=0.02)
                except Exception:
                     # cola llena: descarta
                     pass
        finally:
            cap.release()
    
    @staticmethod
    def processing_process(frame_queue, result_queue, running):
        af = AdaptiveFilter()  # instanciado en el proceso de cómputo
        from queue import Empty
        while running.value == 1:
            try:
                frame = frame_queue.get(timeout=0.05)
                raw_cond = af.analyze_frame(frame)
                cond = af._smooth_conditions(raw_cond)
                out = af.apply_filter(frame, cond)

                data = {
                    "original": frame,
                    "filtered": out,
                    "conditions": cond,
                    "filter_used": af.active_filter
                }
                
                try:
                    result_queue.put(data, timeout=0.02)
                except Exception:
                    pass
            except Empty:
                continue
            except KeyboardInterrupt:
                break
            except Exception:
                # errores puntuales: seguir
                continue

    # UI / Utilidades 
    def _overlay(self, frame, conditions, filter_used):

        img = frame.copy()
        overlay = img.copy()

        # caja
        cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.4, 1
        y = 25
        cv2.putText(img, f"FPS: {self.current_fps:.0f} | {filter_used}", (15, y), font, fs, (255, 255, 255), th)
        y += 15
        b_col = (255, 100, 100) if conditions.get("brightness", 100) < 85 else (255, 255, 255)
        cv2.putText(img, f"Brillo: {conditions.get('brightness', 0):.0f}", (15, y), font, fs, b_col, th)

        # indicadores L/F/R
        yc = 60
        status = {
            "low_light": (100, 255, 255) if conditions.get("low_light") else (60, 60, 60),
            "fog":      (255, 100, 255) if conditions.get("fog")      else (60, 60, 60),
            "rain":     (100, 255, 100) if conditions.get("rain")     else (60, 60, 60),
        }
        cv2.circle(img, (15, yc), 6, status["low_light"], -1); cv2.putText(img, "L", (12, yc+3), font, 0.3, (0,0,0), 1)
        cv2.circle(img, (40, yc), 6, status["fog"], -1);      cv2.putText(img, "F", (37, yc+3), font, 0.3, (0,0,0), 1)
        cv2.circle(img, (65, yc), 6, status["rain"], -1);     cv2.putText(img, "R", (62, yc+3), font, 0.3, (0,0,0), 1)
        return img
    
    def _tick_fps(self):
         
         self.fps_counter += 1
         if self.fps_counter % 30 == 0:
            now = time.time()
            self.current_fps = 30 / (now - self.fps_start_time)
            self.fps_start_time = now
    
    # Loop principal
    def run(self):
        try:
            self.running.value = 1
            p1 = Process(target=self.capture_process, args=(self.camera_id, self.frame_queue, self.running))
            p2 = Process(target=self.processing_process, args=(self.frame_queue, self.result_queue, self.running))
            p1.start(); p2.start()
            self.processes = [p1, p2]

            print("FA iniciado. (q: salir, s: guardar frame)")

            from queue import Empty
            while self.running.value == 1:
                try:
                    data = self.result_queue.get(timeout=0.1)
                    original = data["original"]
                    filtered = data["filtered"]
                    cond = data["conditions"]
                    f_used = data["filter_used"]

                    disp = self._overlay(filtered, cond, f_used)
                    view = np.hstack([original, disp])

                    if view.shape[1] > 1200:
                        scale = 1200 / view.shape[1]
                        view = cv2.resize(view, (int(view.shape[1]*scale), int(view.shape[0]*scale)))

                    cv2.imshow("Filtro Adaptativo - Original vs Filtrado", view)
                    self._tick_fps()

                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'):
                        break
                    elif k == ord('s'):
                        ts = int(time.time())
                        cv2.imwrite(f"resultado_{ts}.jpg", view)
                        print(f"Frame guardado: resultado_{ts}.jpg")
                except Empty:
                    continue
                except KeyboardInterrupt:
                    break
                except Exception:
                    # fallos de UI: continuar
                    continue
        finally:
            self.cleanup()

    def cleanup(self):

        self.running.value = 0
        time.sleep(0.5)
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
        cv2.destroyAllWindows()
        print("Sistema detenido.")
    
def main():

        print("=" * 80)
        print("FILTRO ADAPTATIVO")
        print("=" * 80)
        try:
            app = VideoPipeline(camera_id=0)
            app.run()
        except KeyboardInterrupt:
             print("\nDetenido")
        except Exception as e:
             print(f"Error: {e}")
    
if __name__ == "__main__":
    main()