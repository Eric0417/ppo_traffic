import cv2
import threading
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO


class StreamCamera:
    def __init__(self, url: str, name: str, reopen_delay_sec: float = 1.0):
        self.url = url
        self.name = name
        self.reopen_delay_sec = reopen_delay_sec

        self.cap = self._open_capture()

        self.ret = False
        self.frame = None
        self.lock = threading.Lock()

        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _open_capture(self):
        # Prefer FFMPEG for HLS/m3u8 when available
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        except Exception:
            cap = cv2.VideoCapture(self.url)

        # Reduce latency if backend supports it
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        return cap

    def _reopen(self):
        try:
            self.cap.release()
        except Exception:
            pass
        self.cap = self._open_capture()
        time.sleep(self.reopen_delay_sec)

    def update(self):
        while self.running:
            if not self.cap.isOpened():
                self._reopen()
                continue

            ret, frame = self.cap.read()
            with self.lock:
                self.ret = bool(ret)
                self.frame = frame if ret else None

            if not ret:
                self._reopen()
            else:
                time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            return self.ret, None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass


def main():
    # =========================
    # CONFIG (edit these)
    # =========================
    stream_urls = [
        "https://streaming1.dsatmacau.com/traffic/m2065.m3u8",
        "https://streaming1.dsatmacau.com/traffic/m2259.m3u8",
    ]
    NUM_SLOTS = 2

    MODEL_PATH = "yolov8x.pt"  # set to your real weights path

    # INCLUDE ONLY these classes (COCO: 2=car, 3=motorcycle, 5=bus, 7=truck)
    INCLUDE_CLASSES = [2, 3, 5, 7]

    CONF = 0.35
    IMGSZ = 2048
    TRACKER = "bytetrack.yaml"

    frame_width, frame_height = 640, 360
    line_y = int(frame_height / 1.1)

    JAM_THRESHOLD = 6
    target_fps = 60
    frame_delay = 1.0 / target_fps

    # =========================
    # VALIDATION
    # =========================
    if len(stream_urls) > NUM_SLOTS:
        raise ValueError(f"Max {NUM_SLOTS} URLs, got {len(stream_urls)}")

    p = Path(MODEL_PATH)
    if p.suffix.lower() == ".pt" and not p.exists():
        raise FileNotFoundError(
            f"Model weights not found: {p.resolve()}\n"
            f"Current working directory: {Path.cwd()}\n"
            "Fix by placing the file there or setting MODEL_PATH to an absolute path."
        )

    # =========================
    # LOAD MODELS (one per cam to keep tracker state separated)
    # =========================
    print("Loading models...")
    models = []
    for i in range(len(stream_urls)):
        try:
            m = YOLO(MODEL_PATH)
            models.append(m)
            print(f"[Cam {i+1}] Model loaded: {MODEL_PATH}")
        except Exception as e:
            print(f"[Cam {i+1}] Failed to load model: {e}")
            models.append(None)

    cameras = [StreamCamera(url, f"Cam {i+1}") for i, url in enumerate(stream_urls)]
    track_history = [dict() for _ in range(len(cameras))]
    crossed_counts = [0] * len(cameras)

    print("Start. Press 'q' to quit.")

    try:
        while True:
            start_time = time.time()
            tiles = []

            for slot in range(NUM_SLOTS):
                if slot >= len(cameras):
                    empty = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    cv2.putText(
                        empty, f"Slot {slot+1}: Empty", (50, frame_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2
                    )
                    tiles.append(empty)
                    continue

                cam = cameras[slot]
                ret, frame = cam.get_frame()
                model = models[slot] if slot < len(models) else None

                if not ret or frame is None:
                    black = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    cv2.putText(
                        black, f"{cam.name} Loading...", (50, frame_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )
                    tiles.append(black)
                    continue

                frame = cv2.resize(frame, (frame_width, frame_height))
                cv2.line(frame, (0, line_y), (frame_width, line_y), (255, 0, 0), 2)

                annotated = frame.copy()
                status = "No Model"
                status_color = (0, 0, 255)
                det_count = 0

                if model is not None:
                    try:
                        results = model.track(
                            frame,
                            persist=True,
                            tracker=TRACKER,
                            imgsz=IMGSZ,
                            conf=CONF,
                            classes=INCLUDE_CLASSES,   # <-- INCLUDE ONLY [2,3,5,7]
                            verbose=False,
                        )
                        r0 = results[0]
                        annotated = r0.plot()

                        det_count = 0 if r0.boxes is None else len(r0.boxes)

                        if det_count > JAM_THRESHOLD:
                            status, status_color = "Traffic Jam", (0, 0, 255)
                        else:
                            status, status_color = "Smooth", (0, 255, 0)

                        # Line-cross counting using tracked IDs (if present)
                        if r0.boxes is not None and getattr(r0.boxes, "id", None) is not None:
                            ids = r0.boxes.id.int().cpu().tolist()
                            xywh = r0.boxes.xywh.cpu().tolist()
                            for (x, y, w, h), tid in zip(xywh, ids):
                                cy = int(y)
                                prev = track_history[slot].get(tid)
                                if prev is not None:
                                    crossed = (prev < line_y <= cy) or (prev > line_y >= cy)
                                    if crossed:
                                        crossed_counts[slot] += 1
                                track_history[slot][tid] = cy

                    except Exception as e:
                        annotated = frame.copy()
                        status, status_color = "Track Error", (0, 0, 255)
                        cv2.putText(
                            annotated, f"YOLO error: {type(e).__name__}", (10, frame_height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                        )

                # Overlays
                cv2.putText(
                    annotated, f"{cam.name} | Crossed: {crossed_counts[slot]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
                cv2.putText(
                    annotated, f"Status: {status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
                )
                cv2.putText(
                    annotated, f"Det (only {INCLUDE_CLASSES}): {det_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
                )
                cv2.putText(
                    annotated, f"conf={CONF} imgsz={IMGSZ}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

                tiles.append(annotated)

            row1 = cv2.hconcat([tiles[0], tiles[1], tiles[1]])
            row2 = cv2.hconcat([tiles[0], tiles[1], tiles[1]])
            grid = cv2.vconcat([row1, row2])

            final = cv2.resize(grid, (960, 540))
            cv2.imshow("6-Cam YOLO (2x3) - Include classes only", final)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

    finally:
        print("Closing...")
        for cam in cameras:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()