import os
import cv2
import csv
import time
import math
import threading
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from queue import Queue, Empty
from typing import Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from common.camera import StreamCamera
from common.tracker import clamp_xyxy, iou_xyxy

try:
    import openpyxl  # optional for XLSX
except Exception:
    openpyxl = None


# =========================
# CONFIG
# =========================
MAX_CAMERAS = 60

STREAMS_FILE = r"D:\school\bsd\flow_estimate\streams.txt"   # 放 1~30 行 URL/RTSP/檔案路徑，空行或 # 開頭會忽略
MODEL_PATH = r"D:\school\bsd\flow_estimate\yolo12x.pt"

# 只偵測車輛類（COCO）：2=car, 3=motorcycle, 5=bus, 7=truck
INCLUDE_CLASSES = [2, 3, 5, 7]

DEVICE = 0 if torch.cuda.is_available() else "cpu"
CONF = 0.25
IMGSZ = 640
HALF = True if torch.cuda.is_available() else False

# 30 路建議：每路不必 30FPS，通常 1~3 FPS 就足夠估計塞車
PROC_FPS_PER_CAM = 10.0
PROC_INTERVAL_SEC = 1.0 / PROC_FPS_PER_CAM

BATCH_SIZE = 5  # GPU 夠強可提高；不夠就降低

# 推論用尺寸（先把畫面縮到這個大小再丟進 YOLO，節省算力）
PROC_W, PROC_H = 640, 360

# 顯示馬賽克（30 格）：6x5
GRID_COLS, GRID_ROWS = 6, 10
TILE_W, TILE_H = 320, 180  # 顯示縮圖大小

LINE_Y_RATIO = 0.55  # 計數線 y = PROC_H * ratio
FLOW_WINDOW_SEC = 60.0

# 追蹤（簡單 IoU tracker）
IOU_MATCH_TH = 0.30
TRACK_MAX_AGE = 6          # 幾個「處理幀」沒匹配就刪（以每路 2FPS -> 6 幀約 3 秒）
TRACK_MIN_HITS = 2         # 至少命中幾次才允許計數/納入評分

# 塞車評分參數
O_JAM_REF = 0.35           # 佔用率到 25% 視為接近塞滿（可依視角調）
V_STOP_PX_S = 6.0          # 小於此速度視為停滯（像素/秒）
FREE_SPEED_WINDOW_SEC = 300.0  # 用最近幾分鐘估自由流速度（90 分位）
V_FREE_DEFAULT = 80.0      # 若歷史不足，先用這個（像素/秒）

W_OCC, W_SPD, W_STOP = 0.55, 0.35, 0.10  # 權重總和=1
EMA_ALPHA = 0.05

# 紀錄輸出
LOG_DIR = "logs"
LOG_FORMAT = "csv"  # "csv" / "xlsx" / "both"
LOG_EVERY_SEC = 5.0

SHOW_UI = True
WINDOW_NAME = "Traffic 0-Cam (YOLOv12x) | q=quit"


# =========================
# Utils
# =========================
def load_streams(path: str, max_n: int) -> list[str]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
            if len(out) >= max_n:
                break
    return out


def color_for_id(tid: int) -> tuple[int, int, int]:
    # deterministic BGR
    r = (tid * 37) % 255
    g = (tid * 17) % 255
    b = (tid * 97) % 255
    return int(b), int(g), int(r)


def draw_text_bg(img, text: str, x: int, y: int, scale=0.5, fg=(255, 255, 255), bg=(0, 0, 0), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - base - 6), (x + tw + 6, y + 2), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), font, scale, fg, thickness, cv2.LINE_AA)


# =========================
# Simple IoU Tracker
# =========================

@dataclass
class Track:
    tid: int
    bbox: np.ndarray  # xyxy float32
    conf: float
    cls_id: int
    hits: int = 1
    time_since_update: int = 0
    last_t: float = 0.0
    prev_center: Optional[Tuple[float, float]] = None
    center: Optional[Tuple[float, float]] = None
    speed_px_s: float = 0.0
    speed_ema_px_s: float = 0.0
    crossed: bool = False


class SimpleIOUTracker:
    def __init__(self, iou_th=0.3, max_age=6):
        self.iou_th = float(iou_th)
        self.max_age = int(max_age)
        self._next_id = 1
        self.tracks: list[Track] = []

    @staticmethod
    def _center(b: np.ndarray) -> tuple[float, float]:
        x1, y1, x2, y2 = b
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    def update(self, detections: list[dict], t: float, frame_w: int, frame_h: int) -> list[Track]:
        # age all tracks
        for tr in self.tracks:
            tr.time_since_update += 1

        # Build IoU matrix pairs
        matches = []  # (iou, track_idx, det_idx)
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                iou = iou_xyxy(tr.bbox, det["bbox"])
                if iou >= self.iou_th:
                    matches.append((iou, ti, di))

        matches.sort(key=lambda x: x[0], reverse=True)
        used_tracks = set()
        used_dets = set()

        # Greedy association
        for iou, ti, di in matches:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)

            tr = self.tracks[ti]
            det = detections[di]
            new_bbox = clamp_xyxy(det["bbox"], frame_w, frame_h)

            tr.prev_center = tr.center if tr.center is not None else self._center(tr.bbox)
            tr.bbox = new_bbox
            tr.conf = float(det["conf"])
            tr.cls_id = int(det["cls_id"])
            tr.center = self._center(new_bbox)

            dt = max(1e-3, float(t - tr.last_t)) if tr.last_t > 0 else None
            if dt is not None:
                dx = tr.center[0] - tr.prev_center[0]
                dy = tr.center[1] - tr.prev_center[1]
                spd = math.sqrt(dx * dx + dy * dy) / dt
                tr.speed_px_s = float(spd)
                tr.speed_ema_px_s = 0.6 * tr.speed_ema_px_s + 0.4 * tr.speed_px_s
            tr.last_t = float(t)

            tr.hits += 1
            tr.time_since_update = 0

        # Unmatched detections -> new tracks
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            bbox = clamp_xyxy(det["bbox"], frame_w, frame_h)
            c = self._center(bbox)
            tr = Track(
                tid=self._next_id,
                bbox=bbox,
                conf=float(det["conf"]),
                cls_id=int(det["cls_id"]),
                hits=1,
                time_since_update=0,
                last_t=float(t),
                prev_center=None,
                center=c,
                speed_px_s=0.0,
                speed_ema_px_s=0.0,
                crossed=False,
            )
            self._next_id += 1
            self.tracks.append(tr)

        # Remove dead tracks
        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]

        # Return "current" tracks (updated this tick)
        current = [tr for tr in self.tracks if tr.time_since_update == 0]
        return current


# =========================
# Per-camera state & metrics
# =========================
@dataclass
class CameraState:
    name: str
    tracker: SimpleIOUTracker
    # crossing times for sliding window
    cross_up_times: deque = None
    cross_down_times: deque = None
    cross_up_total: int = 0
    cross_down_total: int = 0

    speed_median_hist: deque = None  # (t, median_speed)
    congestion_ema: float = 0.0

    last_tile: Optional[np.ndarray] = None
    last_proc_t: float = 0.0
    last_log_t: float = 0.0

    last_metrics: dict = None

    def __post_init__(self):
        if self.cross_up_times is None:
            self.cross_up_times = deque()
        if self.cross_down_times is None:
            self.cross_down_times = deque()
        if self.speed_median_hist is None:
            self.speed_median_hist = deque()
        if self.last_metrics is None:
            self.last_metrics = {}


def estimate_v_free(state: CameraState, now_t: float) -> float:
    # drop old
    while state.speed_median_hist and state.speed_median_hist[0][0] < now_t - FREE_SPEED_WINDOW_SEC:
        state.speed_median_hist.popleft()

    if len(state.speed_median_hist) < 20:
        return float(V_FREE_DEFAULT)

    speeds = np.array([s for _, s in state.speed_median_hist], dtype=np.float32)
    v_free = float(np.percentile(speeds, 90))
    return max(float(V_FREE_DEFAULT), v_free)


def update_crossings(state: CameraState, tracks: list[Track], line_y: int, now_t: float):
    # sliding window cleanup
    while state.cross_up_times and state.cross_up_times[0] < now_t - FLOW_WINDOW_SEC:
        state.cross_up_times.popleft()
    while state.cross_down_times and state.cross_down_times[0] < now_t - FLOW_WINDOW_SEC:
        state.cross_down_times.popleft()

    for tr in tracks:
        if tr.hits < TRACK_MIN_HITS:
            continue
        if tr.crossed:
            continue
        if tr.prev_center is None or tr.center is None:
            continue

        y0 = tr.prev_center[1]
        y1 = tr.center[1]

        # Cross line (count once per track)
        if y0 < line_y <= y1:
            state.cross_down_total += 1
            state.cross_down_times.append(now_t)
            tr.crossed = True
        elif y0 > line_y >= y1:
            state.cross_up_total += 1
            state.cross_up_times.append(now_t)
            tr.crossed = True


def compute_metrics(state: CameraState, tracks: list[Track], now_t: float, frame_w: int, frame_h: int) -> dict:
    roi_area = float(frame_w * frame_h)

    # only use stable tracks for metrics
    stable = [tr for tr in tracks if tr.hits >= TRACK_MIN_HITS]

    veh_active = len(stable)
    if veh_active == 0:
        raw = 0.0
        state.congestion_ema = (1.0 - EMA_ALPHA) * state.congestion_ema
        metrics = dict(
            veh_active=0,
            occupancy=0.0,
            median_speed=0.0,
            v_free=estimate_v_free(state, now_t),
            stop_ratio=0.0,
            jam_raw=0.0,
            jam_ema=float(state.congestion_ema),
        )
        return metrics

    # occupancy
    area_sum = 0.0
    speeds = []
    stop_cnt = 0
    for tr in stable:
        x1, y1, x2, y2 = tr.bbox
        area_sum += max(0.0, x2 - x1) * max(0.0, y2 - y1)
        spd = float(tr.speed_ema_px_s)
        speeds.append(spd)
        if spd < V_STOP_PX_S:
            stop_cnt += 1

    occupancy = float(max(0.0, min(1.0, area_sum / roi_area)))
    median_speed = float(np.median(np.array(speeds, dtype=np.float32))) if speeds else 0.0
    stop_ratio = float(stop_cnt / max(1, veh_active))

    # update speed history for v_free estimation
    if median_speed > 1e-3:
        state.speed_median_hist.append((now_t, median_speed))
    v_free = float(estimate_v_free(state, now_t))

    # normalized terms
    o_norm = float(max(0.0, min(1.0, occupancy / max(1e-6, O_JAM_REF))))
    s_norm = 1.0 - float(max(0.0, min(1.0, median_speed / max(1e-6, v_free))))

    raw = float(W_OCC * o_norm + W_SPD * s_norm + W_STOP * stop_ratio)
    raw = float(max(0.0, min(1.0, raw)))

    state.congestion_ema = float(EMA_ALPHA * raw + (1.0 - EMA_ALPHA) * state.congestion_ema)

    metrics = dict(
        veh_active=int(veh_active),
        occupancy=float(occupancy),
        median_speed=float(median_speed),
        v_free=float(v_free),
        stop_ratio=float(stop_ratio),
        jam_raw=float(raw),
        jam_ema=float(state.congestion_ema),
    )
    return metrics


def flow_vpm_from_deque(dq: deque) -> float:
    return float(len(dq) * 60.0 / FLOW_WINDOW_SEC) if FLOW_WINDOW_SEC > 1e-6 else 0.0


# =========================
# Logging (background thread)
# =========================
LOG_HEADER = [
    "ts_iso",
    "cam_idx",
    "cam_name",
    "veh_active",
    "flow_total_vpm",
    "flow_up_vpm",
    "flow_down_vpm",
    "cross_total",
    "cross_up_total",
    "cross_down_total",
    "occupancy",
    "median_speed_px_s",
    "v_free_px_s",
    "stop_ratio",
    "jam_raw",
    "jam_ema",
    "proc_fps_est",
]


class LogWorker(threading.Thread):
    def __init__(self, out_dir: str, fmt: str, queue: Queue):
        super().__init__(daemon=True)
        self.out_dir = out_dir
        self.fmt = fmt.lower()
        self.q = queue
        self.running = True

        self.current_date = None
        self.csv_f = None
        self.csv_writer = None

        self.xlsx_path = None
        self.xlsx_wb = None
        self.xlsx_ws = None
        self.xlsx_rows_since_save = 0
        self.last_xlsx_save_t = time.time()

        os.makedirs(self.out_dir, exist_ok=True)

    def _rotate(self, date_str: str):
        # close existing CSV
        if self.csv_f is not None:
            try:
                self.csv_f.flush()
                self.csv_f.close()
            except Exception:
                pass
        self.csv_f = None
        self.csv_writer = None

        # close/save existing XLSX
        if self.xlsx_wb is not None:
            try:
                self.xlsx_wb.save(self.xlsx_path)
            except Exception:
                pass
        self.xlsx_wb = None
        self.xlsx_ws = None
        self.xlsx_path = None
        self.xlsx_rows_since_save = 0

        self.current_date = date_str

        # open new CSV if needed
        if self.fmt in ("csv", "both"):
            csv_path = os.path.join(self.out_dir, f"traffic_{date_str}.csv")
            is_new = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
            self.csv_f = open(csv_path, "a", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.csv_f, fieldnames=LOG_HEADER)
            if is_new:
                self.csv_writer.writeheader()
                self.csv_f.flush()

        # open new XLSX if needed
        if self.fmt in ("xlsx", "both"):
            if openpyxl is None:
                raise RuntimeError("你選了 xlsx 紀錄，但環境沒有 openpyxl：請 pip install openpyxl")
            self.xlsx_path = os.path.join(self.out_dir, f"traffic_{date_str}.xlsx")
            if os.path.exists(self.xlsx_path):
                self.xlsx_wb = openpyxl.load_workbook(self.xlsx_path)
                self.xlsx_ws = self.xlsx_wb.active
            else:
                self.xlsx_wb = openpyxl.Workbook()
                self.xlsx_ws = self.xlsx_wb.active
                self.xlsx_ws.append(LOG_HEADER)
                self.xlsx_wb.save(self.xlsx_path)

    def _write_row(self, row: dict):
        date_str = row["ts_iso"][:10]  # YYYY-MM-DD
        if self.current_date != date_str:
            self._rotate(date_str)

        if self.csv_writer is not None:
            self.csv_writer.writerow(row)

        if self.xlsx_ws is not None:
            self.xlsx_ws.append([row.get(k, "") for k in LOG_HEADER])
            self.xlsx_rows_since_save += 1

            # periodic save (avoid too frequent)
            now = time.time()
            if self.xlsx_rows_since_save >= 200 or (now - self.last_xlsx_save_t) > 60:
                self.xlsx_wb.save(self.xlsx_path)
                self.xlsx_rows_since_save = 0
                self.last_xlsx_save_t = now

        # flush CSV often for long-running safety
        if self.csv_f is not None:
            self.csv_f.flush()

    def run(self):
        while self.running or not self.q.empty():
            try:
                row = self.q.get(timeout=0.5)
            except Empty:
                continue
            try:
                self._write_row(row)
            except Exception as e:
                # best-effort: don't crash whole program due to logging
                print(f"[LogWorker] write error: {type(e).__name__}: {e}")

        # final save/close
        try:
            if self.csv_f is not None:
                self.csv_f.flush()
                self.csv_f.close()
        except Exception:
            pass
        try:
            if self.xlsx_wb is not None and self.xlsx_path is not None:
                self.xlsx_wb.save(self.xlsx_path)
        except Exception:
            pass

    def stop(self):
        self.running = False


# =========================
# Drawing
# =========================
def annotate(frame: np.ndarray, cam_name: str, tracks: list[Track], metrics: dict, line_y: int,
             flow_up_vpm: float, flow_down_vpm: float) -> np.ndarray:
    img = frame.copy()

    # line
    cv2.line(img, (0, line_y), (img.shape[1], line_y), (255, 0, 0), 2)

    # tracks
    for tr in tracks:
        if tr.hits < TRACK_MIN_HITS:
            continue
        x1, y1, x2, y2 = tr.bbox.astype(int)
        col = color_for_id(tr.tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = f"id:{tr.tid} v:{tr.speed_ema_px_s:.0f}"
        draw_text_bg(img, label, x1, max(18, y1), scale=0.45, fg=(255, 255, 255), bg=col, thickness=1)

    jam = float(metrics.get("jam_ema", 0.0))
    veh = int(metrics.get("veh_active", 0))
    flow_total = flow_up_vpm + flow_down_vpm

    # jam color
    jam_col = (0, int(255 * (1 - jam)), int(255 * jam))  # BGR: green->red

    draw_text_bg(img, f"{cam_name}", 8, 22, scale=0.6, fg=(255, 255, 255), bg=(30, 30, 30), thickness=1)
    draw_text_bg(img, f"Veh:{veh}  Flow:{flow_total:.1f}/min", 8, 46, scale=0.55, fg=(255, 255, 255), bg=(30, 30, 30), thickness=1)
    draw_text_bg(img, f"JamScore:{jam:.2f}", 8, 70, scale=0.55, fg=(255, 255, 255), bg=jam_col, thickness=1)

    # score bar
    bar_w = int((img.shape[1] - 16) * jam)
    cv2.rectangle(img, (8, img.shape[0] - 14), (img.shape[1] - 8, img.shape[0] - 6), (50, 50, 50), -1)
    cv2.rectangle(img, (8, img.shape[0] - 14), (8 + bar_w, img.shape[0] - 6), jam_col, -1)

    return img


# =========================
# Main
# =========================
def main():
    streams = load_streams(STREAMS_FILE, MAX_CAMERAS)
    if len(streams) == 0:
        raise FileNotFoundError(
            f"找不到任何串流。請建立 {STREAMS_FILE}，每行一個 URL/RTSP/檔案路徑（最多 {MAX_CAMERAS} 行）。"
        )

    print(f"Loaded {len(streams)} streams.")
    os.makedirs(LOG_DIR, exist_ok=True)

    # init cameras
    cameras = [StreamCamera(url, f"Cam {i+1:02d}") for i, url in enumerate(streams)]

    # init per-cam state
    states = []
    for i in range(len(cameras)):
        states.append(CameraState(
            name=f"Cam {i+1:02d}",
            tracker=SimpleIOUTracker(iou_th=IOU_MATCH_TH, max_age=TRACK_MAX_AGE)
        ))

    # init model (single)
    print(f"Loading YOLO model: {MODEL_PATH} on device={DEVICE}")
    model = YOLO(MODEL_PATH)
    try:
        model.fuse()
    except Exception:
        pass

    # logging worker
    log_q = Queue(maxsize=20000)
    worker = LogWorker(LOG_DIR, LOG_FORMAT, log_q)
    worker.start()

    line_y = int(PROC_H * LINE_Y_RATIO)

    # prefill tiles
    blank_tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)

    try:
        if SHOW_UI:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            now = time.time()

            # select cams due for processing
            batch_frames = []
            batch_indices = []
            batch_raw_for_annot = []  # processed frames

            for idx in range(len(cameras)):
                st = states[idx]
                if st.last_proc_t > 0 and (now - st.last_proc_t) < PROC_INTERVAL_SEC:
                    continue

                ret, frame = cameras[idx].get_frame()
                if not ret or frame is None:
                    # if no tile yet, show loading
                    if st.last_tile is None:
                        tile = blank_tile.copy()
                        draw_text_bg(tile, f"{st.name} loading...", 10, TILE_H // 2, scale=0.6, fg=(255, 255, 255), bg=(0, 0, 255), thickness=2)
                        st.last_tile = tile
                    continue

                # resize for inference
                proc = cv2.resize(frame, (PROC_W, PROC_H))
                batch_frames.append(proc)
                batch_indices.append(idx)
                batch_raw_for_annot.append(proc)

                st.last_proc_t = now  # reserve slot; if inference fails, next loop will retry anyway

            # run inference in mini-batches
            for b0 in range(0, len(batch_frames), BATCH_SIZE):
                sub_frames = batch_frames[b0:b0 + BATCH_SIZE]
                sub_indices = batch_indices[b0:b0 + BATCH_SIZE]

                if len(sub_frames) == 0:
                    continue

                results = model.predict(
                    sub_frames,
                    imgsz=IMGSZ,
                    conf=CONF,
                    classes=INCLUDE_CLASSES,
                    device=DEVICE,
                    half=HALF,
                    verbose=False,
                )

                for res, cam_idx, proc_frame in zip(results, sub_indices, sub_frames):
                    st = states[cam_idx]
                    t_cam = time.time()

                    # extract detections
                    dets = []
                    boxes = res.boxes
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                        confs = boxes.conf.detach().cpu().numpy()
                        clss = boxes.cls.detach().cpu().numpy().astype(int)
                        for i in range(len(xyxy)):
                            dets.append({
                                "bbox": xyxy[i].astype(np.float32),
                                "conf": float(confs[i]),
                                "cls_id": int(clss[i]),
                            })

                    # tracking
                    tracks = st.tracker.update(dets, t_cam, PROC_W, PROC_H)

                    # crossings & flow
                    update_crossings(st, tracks, line_y=line_y, now_t=t_cam)
                    flow_up_vpm = flow_vpm_from_deque(st.cross_up_times)
                    flow_down_vpm = flow_vpm_from_deque(st.cross_down_times)

                    # metrics & score
                    metrics = compute_metrics(st, tracks, now_t=t_cam, frame_w=PROC_W, frame_h=PROC_H)
                    st.last_metrics = metrics

                    # annotate & tile
                    ann = annotate(proc_frame, st.name, tracks, metrics, line_y, flow_up_vpm, flow_down_vpm)
                    tile = cv2.resize(ann, (TILE_W, TILE_H))
                    st.last_tile = tile

                    # logging (rate-limited)
                    if (t_cam - st.last_log_t) >= LOG_EVERY_SEC:
                        st.last_log_t = t_cam
                        ts_iso = datetime.fromtimestamp(t_cam).isoformat(timespec="seconds")

                        cross_up = int(st.cross_up_total)
                        cross_down = int(st.cross_down_total)
                        cross_total = cross_up + cross_down

                        # per-cam proc fps estimate
                        # (rough; uses PROC_INTERVAL_SEC as target)
                        proc_fps_est = float(PROC_FPS_PER_CAM)

                        row = {
                            "ts_iso": ts_iso,
                            "cam_idx": int(cam_idx + 1),
                            "cam_name": st.name,
                            "veh_active": int(metrics["veh_active"]),
                            "flow_total_vpm": float(flow_up_vpm + flow_down_vpm),
                            "flow_up_vpm": float(flow_up_vpm),
                            "flow_down_vpm": float(flow_down_vpm),
                            "cross_total": int(cross_total),
                            "cross_up_total": int(cross_up),
                            "cross_down_total": int(cross_down),
                            "occupancy": float(metrics["occupancy"]),
                            "median_speed_px_s": float(metrics["median_speed"]),
                            "v_free_px_s": float(metrics["v_free"]),
                            "stop_ratio": float(metrics["stop_ratio"]),
                            "jam_raw": float(metrics["jam_raw"]),
                            "jam_ema": float(metrics["jam_ema"]),
                            "proc_fps_est": float(proc_fps_est),
                        }
                        try:
                            log_q.put_nowait(row)
                        except Exception:
                            pass

            # build mosaic
            if SHOW_UI:
                grid = np.zeros((GRID_ROWS * TILE_H, GRID_COLS * TILE_W, 3), dtype=np.uint8)
                for i in range(GRID_ROWS * GRID_COLS):
                    r = i // GRID_COLS
                    c = i % GRID_COLS
                    y0, y1 = r * TILE_H, (r + 1) * TILE_H
                    x0, x1 = c * TILE_W, (c + 1) * TILE_W

                    if i < len(states) and states[i].last_tile is not None:
                        grid[y0:y1, x0:x1] = states[i].last_tile
                    else:
                        grid[y0:y1, x0:x1] = blank_tile

                cv2.imshow(WINDOW_NAME, grid)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            time.sleep(0.005)

    finally:
        print("Stopping...")
        for cam in cameras:
            cam.stop()

        worker.stop()
        worker.join(timeout=3.0)

        if SHOW_UI:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()