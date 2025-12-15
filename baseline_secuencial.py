import cv2
import numpy as np
import time
from collections import deque

# Mismo Filtro Adaptativo pero sin paralelismo

class AdaptiveFilter:
    def __init__(self):
        # Guardamos un pequeño historial para suavizar resultados
        self.history = deque(maxlen=3)
        self.times = []
        self.active_filter = "none"

        # Inicializamos los kernels que usan los filtros
        self._init_kernels()

        # Resolución más baja para analizar más rápido
        self.resolution = (320, 240)

    def _init_kernels(self):
        self.edge_kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        self.gaussian_kernel = cv2.getGaussianKernel(3, 0.5)
        print("Kernels inicializados")

    def analyze_frame(self, frame):
        resized = cv2.resize(frame, self.resolution)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Solo analizamos la zona central 
        h, w = gray.shape
        center = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]

        # Calculamos brillo y ruido
        brightness = float(np.mean(gray))
        center_brightness = float(np.mean(center))
        noise_level = float(np.std(center))

        # Analizamos el histograma de intensidades
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / (hist.sum() if hist.sum() > 0 else 1.0)
        dark_pixels = float(np.sum(hist_norm[:64]))      # Píxeles muy oscuros
        bright_pixels = float(np.sum(hist_norm[192:]))   # Píxeles muy claros

        # Detección de baja iluminación (mezcla de varios indicadores)
        low_light_flags = [
            brightness < 60,
            center_brightness < 50,
            dark_pixels > 0.80,
            bright_pixels < 0.03,
            noise_level > 25
        ]
        low_light_conf = sum(low_light_flags) / len(low_light_flags)
        if brightness < 60 and dark_pixels > 0.7:
            low_light_conf = min(low_light_conf + 0.3, 1.0)
        low_light = (brightness < 85) or (low_light_conf >= 0.6)

        # Detección de niebla (baja nitidez y saturación)
        lap_var = float(cv2.Laplacian(center, cv2.CV_64F).var())
        fog = False
        if brightness > 70:
            sobelx = cv2.Sobel(center, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(center, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mean = float(np.mean(np.abs(sobelx) + np.abs(sobely)))
            local_std = float(np.std(center))
            avg_sat = float(np.mean(hsv[:, :, 1]))
            fog_flags = [
                lap_var < 500,
                sobel_mean < 20,
                local_std < 35,
                avg_sat < 120,
                brightness > 50
            ]
            fog_conf = sum(fog_flags) / len(fog_flags)
            fog = (fog_conf >= 0.7) and (brightness >= 75) and (low_light_conf < 0.2)
        else:
            fog_conf = 0.0

        # Detección de lluvia (busca líneas finas casi verticales)
        rain = False
        rain_conf = 0.0
        if brightness > 60 and noise_level < 20 and low_light_conf < 0.4:
            gray_blur = cv2.medianBlur(gray, 3)
            edges = cv2.Canny(gray_blur, 40, 120)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength=20, maxLineGap=5)

            vertical = 0
            diagonal = 0
            if lines is not None:
                for (x1, y1, x2, y2) in lines[:, 0]:
                    length = np.hypot(x2 - x1, y2 - y1)
                    ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    if 85 <= ang <= 95 and length > 20:
                        vertical += 1
                    elif 70 <= ang <= 110 and length > 25:
                        diagonal += 1

            total_rel = vertical + diagonal
            density = (total_rel / len(lines)) if (lines is not None and len(lines) > 0) else 0.0
            rain_flags = [
                vertical >= 12,
                density > 0.6,
                brightness > 60,
                noise_level < 15,
                low_light_conf < 0.3
            ]
            if all(rain_flags):
                rain = True
                rain_conf = 0.9

        return {
            'low_light': bool(low_light),
            'fog': bool(fog),
            'rain': bool(rain),
            'brightness_value': brightness,
            'blur_value': lap_var,
            'low_light_confidence': low_light_conf,
            'fog_confidence': fog_conf,
            'rain_confidence': rain_conf
        }

    def _smooth_conditions(self, conditions):
        self.history.append(conditions)
        if len(self.history) < 2:
            return conditions

        out = {}
        for key in ("low_light", "fog", "rain"):
            votes = sum(1 for c in self.history if c.get(key))
            out[key] = votes >= (len(self.history) // 2)

        out["brightness_value"] = conditions.get("brightness_value", 0.0)
        out["blur_value"] = conditions.get("blur_value", 0.0)
        return out

    def _apply_lime(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        img = small.astype(np.float32) / 255.0
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        illumination = cv2.bilateralFilter(gray, 5, 80, 80).astype(np.float32) / 255.0
        illumination = np.clip(illumination, 0.2, 1.0)
        out = np.zeros_like(img)
        for i in range(3):
            out[:, :, i] = np.power(img[:, :, i] / illumination, 0.6)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        return cv2.resize(out, (w, h))

    def _apply_fvr(self, frame):

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * 0.6), int(h * 0.6)))
        img = small.astype(np.float64) / 255.0
        airlight = np.percentile(small, 95, axis=(0, 1))
        dark_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dark = cv2.morphologyEx(dark_channel, cv2.MORPH_ERODE, kernel)
        transmission = 1 - 0.95 * dark / (airlight.max() / 255.0)
        transmission = np.maximum(transmission, 0.2)
        res = np.zeros_like(img)
        for i in range(3):
            res[:, :, i] = (img[:, :, i] - airlight[i] / 255.0) / transmission + airlight[i] / 255.0
        res = np.clip(res * 255, 0, 255).astype(np.uint8)
        return cv2.resize(res, (w, h))

    def _apply_arr(self, frame):
        # Elimina gotas de lluvia usando inpainting.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        rain_mask = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_v)
        _, rain_mask = cv2.threshold(rain_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        rain_mask = cv2.dilate(rain_mask, kernel_d, iterations=1)
        clean = cv2.inpaint(frame, rain_mask, 3, cv2.INPAINT_TELEA)
        return cv2.bilateralFilter(clean, 5, 50, 50)

    def apply_filter(self, frame, conditions):
        # Elige el filtro adecuado según la condición detectada
        if conditions["rain"]:
            output = self._apply_arr(frame)
            self.active_filter = "ARR (Rain Removal)"
        elif conditions["fog"]:
            output = self._apply_fvr(frame)
            self.active_filter = "FVR (Fog Removal)"
        elif conditions["low_light"]:
            output = self._apply_lime(frame)
            self.active_filter = "LIME (Low Light)"
        else:
            output = frame.copy()
            self.active_filter = "Sin filtro"
        return output


class SequentialBaseline:
    # Ejecuta todo el pipeline de forma secuencial, sin procesos paralelos.

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.af = AdaptiveFilter()
        self.fps_counter = 0
        self.current_fps = 0.0
        self.fps_start_time = time.time()

    def _init_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")

    def _tick_fps(self):
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            now = time.time()
            self.current_fps = 30 / (now - self.fps_start_time)
            self.fps_start_time = now

    def initialize_camera(self):
        """Alias para _init_camera, usado por medir_rendimiento.py"""
        self._init_camera()

    def cleanup(self):
        """Libera recursos de la cámara"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def process_frame_sequential(self, frame):
        """
        Procesa un frame de forma secuencial y mide el tiempo.
        Returns:
            tuple: (frame_filtrado, condiciones, tiempo_procesamiento)
        """
        start_time = time.time()
        
        # Análisis y filtrado secuencial
        cond_raw = self.af.analyze_frame(frame)
        cond = self.af._smooth_conditions(cond_raw)
        filtered = self.af.apply_filter(frame, cond)
        
        proc_time = time.time() - start_time
        return filtered, cond, proc_time

    def _overlay(self, frame, conditions, filter_used):
        img = frame.copy()
        box = img.copy()
        cv2.rectangle(box, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.addWeighted(box, 0.6, img, 0.4, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.4, 1
        y = 25
        cv2.putText(img, f"FPS: {self.current_fps:.0f} | {filter_used}", (15, y), font, fs, (255, 255, 255), th)
        y += 15
        val_bright = conditions.get("brightness_value", 0.0)
        col = (255, 100, 100) if val_bright < 85 else (255, 255, 255)
        cv2.putText(img, f"Brillo: {val_bright:.0f}", (15, y), font, fs, col, th)

        yc = 60
        dot = {
            "low_light": (100, 255, 255) if conditions.get("low_light") else (60, 60, 60),
            "fog":      (255, 100, 255) if conditions.get("fog") else (60, 60, 60),
            "rain":     (100, 255, 100) if conditions.get("rain") else (60, 60, 60),
        }
        cv2.circle(img, (15, yc), 6, dot["low_light"], -1); cv2.putText(img, "L", (12, yc+3), font, 0.3, (0,0,0), 1)
        cv2.circle(img, (40, yc), 6, dot["fog"], -1);      cv2.putText(img, "F", (37, yc+3), font, 0.3, (0,0,0), 1)
        cv2.circle(img, (65, yc), 6, dot["rain"], -1);     cv2.putText(img, "R", (62, yc+3), font, 0.3, (0,0,0), 1)
        return img

    def run(self):
        self._init_camera()
        print("Ejecutando versión secuencial. (q: salir, s: guardar frame)")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                # Flujo principal (todo en el mismo hilo)
                cond_raw = self.af.analyze_frame(frame)
                cond = self.af._smooth_conditions(cond_raw)
                filtered = self.af.apply_filter(frame, cond)

                disp = self._overlay(filtered, cond, self.af.active_filter)
                view = np.hstack([frame, disp])

                if view.shape[1] > 1200:
                    scale = 1200 / view.shape[1]
                    view = cv2.resize(view, (int(view.shape[1]*scale), int(view.shape[0]*scale)))

                cv2.imshow("Versión Secuencial - Original vs Filtrado", view)
                self._tick_fps()

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('s'):
                    ts = int(time.time())
                    cv2.imwrite(f"resultado_seq_{ts}.jpg", view)
                    print(f"Frame guardado: resultado_seq_{ts}.jpg")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Sistema detenido .")


def main():
    print("=" * 80)
    print("EJECUCIÓN SECUENCIAL DEL FILTRO ")
    print("=" * 80)
    try:
        app = SequentialBaseline(camera_id=0)
        app.run()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
