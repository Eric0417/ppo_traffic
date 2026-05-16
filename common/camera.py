import time
import threading

import cv2


class StreamCamera:
    def __init__(self, url: str, name: str, reopen_delay_sec: float = 1.0, read_sleep_sec: float = 0.001):
        self.url = url
        self.name = name
        self.reopen_delay_sec = reopen_delay_sec
        self.read_sleep_sec = read_sleep_sec

        self.cap = self._open_capture()

        self.ret = False
        self.frame = None
        self.lock = threading.Lock()

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open_capture(self):
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        except Exception:
            cap = cv2.VideoCapture(self.url)

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

    def _update(self):
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
                time.sleep(self.read_sleep_sec)

    def get_frame(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

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
