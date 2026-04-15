import os
import cv2
import time
import math
import threading
import csv
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from queue import Queue, Empty
from typing import Optional, Tuple
import copy
import json
import requests

import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# 進階路網分析
try:
    import osmnx as ox
    ox.settings.log_console = False
    ox.settings.use_cache = True
except ImportError:
    ox = None

# =========================
# CONFIG (基礎設定)
# =========================
MAX_CAMERAS = 60
CLOUD_API_URL = "https://macau-traffic.onrender.com/api/update" 

STREAMS_FILE = r"D:\school\bsd\2\streams.txt"
MODEL_PATH = r"D:\school\bsd\2\yolo12l.pt"
FONT_PATH = r"C:\Windows\Fonts\msjh.ttc" # Windows 內建微軟正黑體
CSV_LOG_FILE = r"D:\school\bsd\log\traffic_jam_log.csv" # 🚀 CSV 數據紀錄檔路徑

# 只偵測車輛類（COCO）：2=car, 3=motorcycle, 5=bus, 7=truck
INCLUDE_CLASSES = [2, 3, 5, 7]

DEVICE = 0 if torch.cuda.is_available() else "cpu"
CONF = 0.25
IMGSZ = 640
PROC_FPS_PER_CAM = 5.0
PROC_INTERVAL_SEC = 1.0 / PROC_FPS_PER_CAM
PROC_W, PROC_H = 640, 360

# --- 鷹眼模式與違停過濾設定 ---
PARKED_TIME_SEC = 120.0     # 靜止超過此秒數視為路邊停車
STATIONARY_SPEED_TH = 2.0   # 速度低於此值(px/s)視為靜止

# 🌟 [修改] 針對廣角/遠景鏡頭設定動態速度與「場地容量係數」
CAM_OVERRIDES = {
    "亞馬喇前地之圓形地方向": {
        "proc_size": None, 
        "imgsz": 1280, 
        "conf": 0.15,
        "speed_scale": 0.2,    # 像素移動速度標準降為 20%
        "capacity_scale": 4.0  # 🌟 圓形地容量是普通道路的 4 倍！(48台才算塞)
    },
    "亞馬喇前地巴士站": {
        "proc_size": None,
        "imgsz": 1280,
        "conf": 0.15,
        "speed_scale": 0.3,
        "capacity_scale": 3.0  # 🌟 巴士站容量是普通道路的 3 倍！(36台才算塞)
    }
}

GRID_COLS, GRID_ROWS = 6, 10
TILE_W, TILE_H = 320, 180  
LINE_Y_RATIO = 0.55 # 虛擬計數線的位置 (Y軸 55%)
SMOOTH_WINDOW_SIZE = 300 
TRACK_MIN_HITS = 1
SHOW_UI = True
WINDOW_NAME = "Macau Traffic Real-time | q=quit"

# =========================
# 街道座標設定
# =========================
MACAU_STREETS = [
    "提督馬路與高士德大馬路交界（向高士德大馬路方向）", "巴坡沙大馬路與青洲大馬路交界", "提督馬路與罅些喇海軍上將巷交界", "提督馬路與高士德大馬路交界", 
    "沙梨頭海邊街與爹美刁施拿地大馬路交界", "勞動節大馬路", "慕拉士大馬路", "友誼橋大馬路(向氹仔方向)", "友誼大馬路(向馬揸度博士大馬路方向)", "馬揸度博士大馬路", 
    "馬揸度博士大馬路(近勞工局)", "馬場北大馬路與馬場東大馬路交界", "友誼圓形地與友誼橋大馬路交界(向港珠澳大橋入口方向)", "美副將大馬路與俾利喇街交界", 
    "東北大馬路與黑沙環中街交界向友誼圓形地方向", "黑沙環新街", "馬交石斜坡與俾利喇街交界", "A2橋澳門出口", "友誼大馬路", "馬六甲街停車場", "松山隧道羅理基方向", 
    "羅理基博士大馬路行車隧道往松山隧道方向", "宋玉生廣場", "捐血中心", "孫逸仙大馬路與城市日大馬路交界", "孫逸仙大馬路(近終審法院前地)", 
    "西灣湖廣場(向孫逸仙大馬路)", "西灣湖廣場(向西灣湖景大馬路)", "高士德與俾利喇街交界", "華士古停車場", "美副將大馬路與連勝馬路交界", "松山隧道高士德方向", 
    "美副將大馬路與士多鳥拜斯大馬路交界", "高偉樂街與荷蘭園大馬路交界", "水坑尾", "沙梨頭海邊街", "亞馬喇前地之圓形地方向", "亞馬喇前地", "亞馬喇前地巴士站", 
    "南灣大馬路與區華利前地交界", "南灣大馬路與殷皇子大馬路(向八角亭方向)", "南灣大馬路與殷皇子大馬路(向殷皇子大馬路方向)", "巴素打爾古街近栢港停車場出口(向火船頭街方向)", 
    "新馬路(近議事亭前地)向南灣大馬路方向", "殷皇子大馬路與約翰四世大馬路交界", "殷皇子大馬路與葡京路交界", "水坑尾街與南灣大馬路交界路口", "沙梨頭海邊街與林茂巷交界", 
    "沙梨頭海邊街與魚鰓巷交界(向新馬路方向)", "爹美刁斯拿地大馬路近栢港停車場", "爹美刁施拿地大馬路與魚鱗巷交界(向十六浦方向)", "新馬路與巴素打爾古街交界", 
    "新馬路與南灣大馬路交界", "比厘喇馬忌士街與貨倉巷交界(向十六浦方向)", "比厘喇馬忌士街與馬博士巷交界", "河邊新街與比厘喇馬忌士街交界(向媽閣廟方向)", 
    "河邊新街與航海學校街交界(向媽閣方向)", "河邊新街與鹽巷交界（向媽閣方向）", "火船頭街近11號碼頭（向巴素打爾古街方向)", "火船頭街（近11號碼頭）（向河邊新街方向）"
]

ORIGINAL_COORDS = {
    "提督馬路與高士德大馬路交界（向高士德大馬路方向）": (22.20533, 113.54436), "巴坡沙大馬路與青洲大馬路交界": (22.21027, 113.54734),
    "提督馬路與罅些喇海軍上將巷交界": (22.20401, 113.54255), "提督馬路與高士德大馬路交界": (22.20525, 113.54427),
    "沙梨頭海邊街與爹美刁施拿地大馬路交界": (22.20248, 113.53801), "勞動節大馬路": (22.21059, 113.55409),
    "慕拉士大馬路": (22.20665, 113.55256), "友誼橋大馬路(向氹仔方向)": (22.20866, 113.56081),
    "友誼大馬路(向馬揸度博士大馬路方向)": (22.20474, 113.56060), "馬揸度博士大馬路": (22.20563, 113.55812),
    "馬揸度博士大馬路(近勞工局)": (22.20573, 113.55884), "馬場北大馬路與馬場東大馬路交界": (22.21385, 113.55444),
    "友誼圓形地與友誼橋大馬路交界(向港珠澳大橋入口方向)": (22.21175, 113.55951), "美副將大馬路與俾利喇街交界": (22.20496, 113.54877),
    "東北大馬路與黑沙環中街交界向友誼圓形地方向": (22.20938, 113.55765), "黑沙環新街": (22.20613, 113.55738),
    "馬交石斜坡與俾利喇街交界": (22.20669, 113.55032), "A2橋澳門出口": (22.212000, 113.555000), 
    "友誼大馬路": (22.19077, 113.54870), "馬六甲街停車場": (22.19606, 113.55367),
    "松山隧道羅理基方向": (22.19674, 113.55192), "羅理基博士大馬路行車隧道往松山隧道方向": (22.19619, 113.55196),
    "宋玉生廣場": (22.19006, 113.55043), "捐血中心": (22.18946, 113.54977),
    "孫逸仙大馬路與城市日大馬路交界": (22.18585, 113.54943), "孫逸仙大馬路(近終審法院前地)": (22.18239, 113.54111),
    "西灣湖廣場(向孫逸仙大馬路)": (22.18146, 113.53839), "西灣湖廣場(向西灣湖景大馬路)": (22.18044, 113.53762),
    "高士德與俾利喇街交界": (22.20290, 113.54709), "華士古停車場": (22.19639, 113.54650),
    "美副將大馬路與連勝馬路交界": (22.20625, 113.54715), "松山隧道高士德方向": (22.19889, 113.55073),
    "美副將大馬路與士多鳥拜斯大馬路交界": (22.20191, 113.55247), "高偉樂街與荷蘭園大馬路交界": (22.19768, 113.54638),
    "水坑尾": (22.19405, 113.54369), "沙梨頭海邊街": (22.20314, 113.54033),
    "亞馬喇前地之圓形地方向": (22.18895, 113.54317), "亞馬喇前地": (22.18990, 113.54339),
    "亞馬喇前地巴士站": (22.18944, 113.54336), "南灣大馬路與區華利前地交界": (22.19105, 113.53927),
    "南灣大馬路與殷皇子大馬路(向八角亭方向)": (22.19224, 113.54096), "南灣大馬路與殷皇子大馬路(向殷皇子大馬路方向)": (22.19219, 113.54089),
    "巴素打爾古街近栢港停車場出口(向火船頭街方向)": (22.19765, 113.53669), "新馬路(近議事亭前地)向南灣大馬路方向": (22.19234, 113.54081),
    "殷皇子大馬路與約翰四世大馬路交界": (22.19111, 113.54202), "殷皇子大馬路與葡京路交界": (22.18993, 113.54341),
    "水坑尾街與南灣大馬路交界路口": (22.19273, 113.54340), "沙梨頭海邊街與林茂巷交界": (22.20250, 113.53806),
    "沙梨頭海邊街與魚鰓巷交界(向新馬路方向)": (22.20018, 113.53728), "爹美刁斯拿地大馬路近栢港停車場": (22.19829, 113.53657),
    "爹美刁施拿地大馬路與魚鱗巷交界(向十六浦方向)": (22.20174, 113.53727), "新馬路與巴素打爾古街交界": (22.19642, 113.53648),
    "新馬路與南灣大馬路交界": (22.19225, 113.54085), "比厘喇馬忌士街與貨倉巷交界(向十六浦方向)": (22.19108, 113.53459),
    "比厘喇馬忌士街與馬博士巷交界": (22.19218, 113.53400), "河邊新街與比厘喇馬忌士街交界(向媽閣廟方向)": (22.18947, 113.53268),
    "河邊新街與航海學校街交界(向媽閣方向)": (22.18746, 113.53120), "河邊新街與鹽巷交界（向媽閣方向）": (22.19011, 113.53309),
    "火船頭街近11號碼頭（向巴素打爾古街方向)": (22.19630, 113.53647), "火船頭街（近11號碼頭）（向河邊新街方向）": (22.19319, 113.53520)
}

def resolve_overlaps(coords_dict, min_dist=0.00015):
    coords = {k: list(v) for k, v in coords_dict.items()}
    keys = list(coords.keys())
    for _ in range(15):
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                k1, k2 = keys[i], keys[j]
                lat1, lon1 = coords[k1]; lat2, lon2 = coords[k2]
                dist = math.hypot(lat1 - lat2, lon1 - lon2)
                if dist < min_dist:
                    angle = math.atan2(lat1 - lat2, lon1 - lon2) if dist != 0 else i * 0.5
                    push = (min_dist - (dist if dist != 0 else 0.00001)) / 2.0
                    coords[k1][0] += push * math.sin(angle); coords[k1][1] += push * math.cos(angle)
                    coords[k2][0] -= push * math.sin(angle); coords[k2][1] -= push * math.cos(angle)
    return {k: tuple(v) for k, v in coords.items()}

STREET_COORDS = resolve_overlaps(ORIGINAL_COORDS)

# =========================
# 雲端資料推播演算法
# =========================
MACAU_GRAPH = None
def preload_macau_graph():
    global MACAU_GRAPH
    if ox is None: return
    try:
        MACAU_GRAPH = ox.graph_from_bbox(bbox=(113.525, 22.178, 113.565, 22.217), network_type='drive')
        print("✅ 澳門路網載入成功！")
    except Exception as e: 
        print(f"❌ 路網載入失敗: {e}")

def run_heat_diffusion(G, source_scores, iterations=40):
    scores = {n: None for n in G.nodes()}
    for n, s in source_scores.items(): scores[n] = s
    for _ in range(iterations):
        new_scores = scores.copy()
        for n in G.nodes():
            if n in source_scores: continue 
            neighbors = list(G.successors(n)) + list(G.predecessors(n))
            vals = [scores[nb] for nb in neighbors if scores[nb] is not None]
            if vals: new_scores[n] = sum(vals) / len(vals)
        scores = new_scores
    for n in G.nodes():
        if scores[n] is None: scores[n] = 0.0
    return scores

def post_macau_traffic_data(states, cloud_url):
    if MACAU_GRAPH is None: return
    try:
        current_scores = {st.name: (st.last_metrics.get("jam_avg", 0.0) if st.last_metrics else 0.0) for st in states}
        def get_color(score):
            if score < 0.3: return "#28a745"
            elif score < 0.75: return "#ffc107"
            else: return "#dc3545"

        camera_node_scores = {}
        for name, coord in STREET_COORDS.items():
            if name in current_scores:
                nearest_node = ox.distance.nearest_nodes(MACAU_GRAPH, X=coord[1], Y=coord[0])
                camera_node_scores[nearest_node] = current_scores[name]

        node_scores = run_heat_diffusion(MACAU_GRAPH, camera_node_scores)
        features = []
        for u, v, k, data in MACAU_GRAPH.edges(keys=True, data=True):
            sc = (node_scores[u] + node_scores[v]) / 2.0
            coords = [[lon, lat] for lon, lat in data['geometry'].coords] if 'geometry' in data else [[MACAU_GRAPH.nodes[u]['x'], MACAU_GRAPH.nodes[u]['y']], [MACAU_GRAPH.nodes[v]['x'], MACAU_GRAPH.nodes[v]['y']]]
            features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}, "properties": {"type": "street", "score": round(sc, 3), "color": get_color(sc)}})

        for st in states:
            if st.name not in STREET_COORDS: continue
            lat, lon = STREET_COORDS[st.name]
            score = current_scores.get(st.name, 0.0)
            features.append({
                "type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, 
                "properties": {
                    "type": "camera", 
                    "name": st.name, 
                    "score": round(score, 3), 
                    "color": get_color(score), 
                    "veh_active": st.last_metrics.get("veh_active", 0) if st.last_metrics else 0, 
                    "flow_total": st.last_metrics.get("flow_total_vpm", 0) if st.last_metrics else 0,
                    "stream_url": st.stream_url 
                }
            })
        payload = {"type": "FeatureCollection", "features": features, "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        requests.post(cloud_url, json=payload, timeout=2.0)
    except Exception as e:
        print(f"⚠️ 雲端推播背景作業發生錯誤: {e}");print(time.ctime(time.time()))

# =========================
# CSV 數據紀錄演算法
# =========================
def log_jam_to_csv(states, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                headers = ["Timestamp"] + [st.name for st in states]
                writer.writerow(headers)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = [timestamp] + [round(st.last_metrics.get("jam_avg", 0.0), 3) for st in states]
            
            writer.writerow(row)
    except Exception as e:
        print(f"⚠️ CSV 寫入失敗: {e}")

# =========================
# 追蹤與影像處理類
# =========================
@dataclass
class Track:
    tid: int; bbox: np.ndarray; conf: float; cls_id: int
    hits: int = 1; time_since_update: int = 0; last_t: float = 0.0
    center: Optional[Tuple[float, float]] = None; speed_ema_px_s: float = 0.0
    stationary_time: float = 0.0 
    prev_center: Optional[Tuple[float, float]] = None
    counted: bool = False

def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1); union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0

class SimpleIOUTracker:
    def __init__(self, iou_th=0.3, max_age=6):
        self.iou_th, self.max_age, self._next_id = iou_th, max_age, 1
        self.tracks: list[Track] = []

    def update(self, detections, t, fw, fh, stationary_th=2.0):
        for tr in self.tracks: tr.time_since_update += 1
        matches = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                iou = iou_xyxy(tr.bbox, det["bbox"])
                if iou >= self.iou_th: matches.append((iou, ti, di))
        matches.sort(key=lambda x: x[0], reverse=True)
        u_tr, u_det = set(), set()
        
        for iou, ti, di in matches:
            if ti in u_tr or di in u_det: continue
            u_tr.add(ti); u_det.add(di)
            tr, det = self.tracks[ti], detections[di]
            prev_c = tr.center if tr.center else (float((tr.bbox[0]+tr.bbox[2])/2), float((tr.bbox[1]+tr.bbox[3])/2))
            tr.bbox, tr.center = det["bbox"], (float((det["bbox"][0]+det["bbox"][2])/2), float((det["bbox"][1]+det["bbox"][3])/2))
            tr.prev_center = prev_c
            
            dt = max(1e-3, t - tr.last_t)
            speed = math.hypot(tr.center[0]-prev_c[0], tr.center[1]-prev_c[1])/dt
            tr.speed_ema_px_s = 0.6 * tr.speed_ema_px_s + 0.4 * speed
            
            if tr.speed_ema_px_s < stationary_th:
                tr.stationary_time += dt
            else:
                tr.stationary_time = 0.0 
            tr.last_t, tr.hits, tr.time_since_update = float(t), tr.hits + 1, 0
            
        for di, det in enumerate(detections):
            if di not in u_det:
                self.tracks.append(Track(self._next_id, det["bbox"], det["conf"], det["cls_id"], last_t=t, center=(float((det["bbox"][0]+det["bbox"][2])/2), float((det["bbox"][1]+det["bbox"][3])/2))))
                self._next_id += 1
        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]
        return [tr for tr in self.tracks if tr.time_since_update == 0]

class StreamCamera:
    def __init__(self, url: str, name: str):
        self.url, self.name = url, name; self.cap = cv2.VideoCapture(url)
        self.ret, self.frame = False, None; self.lock = threading.Lock()
        self.running = True; threading.Thread(target=self._update, daemon=True).start()
    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock: self.ret, self.frame = ret, frame
            if not ret: time.sleep(1); self.cap = cv2.VideoCapture(self.url)
    def get_frame(self):
        with self.lock: return self.ret, (self.frame.copy() if self.ret and self.frame is not None else None)
    def stop(self): self.running = False

@dataclass
class CameraState:
    name: str; tracker: SimpleIOUTracker
    stream_url: str = "" 
    jam_history: deque = None; veh_history: deque = None
    last_tile: np.ndarray = None; last_proc_t: float = 0.0; last_metrics: dict = None
    flow_timestamps: deque = None 
    bg_subtractor: any = None  
    use_motion: bool = False   

    def __post_init__(self):
        self.jam_history = deque(maxlen=SMOOTH_WINDOW_SIZE)
        self.veh_history = deque(maxlen=600) 
        self.flow_timestamps = deque() 
        self.last_metrics = {"veh_active": 0, "jam_avg": 0.0, "flow_total_vpm": 0}
        
        # 開啟亞馬喇前地的動態捕捉
        if self.name in ["亞馬喇前地之圓形地方向", "亞馬喇前地巴士站"]:
            self.use_motion = True
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=30, detectShadows=False)

def draw_chinese_text_bg(img, text, x, y, font_size=16, fg=(255, 255, 255), bg=(0, 0, 0)):
    try: font = ImageFont.truetype(FONT_PATH, font_size)
    except: font = ImageFont.load_default()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=bg)
    draw.text((x, y), text, font=font, fill=fg)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# =========================
# Main 核心迴圈
# =========================
def main():
    if not os.path.exists(STREAMS_FILE): 
        print(f"找不到串流檔案: {STREAMS_FILE}")
        return
        
    with open(STREAMS_FILE, "r", encoding="utf-8") as f:
        streams = [line.strip() for line in f if line.strip() and not line.startswith("#")][:MAX_CAMERAS]
    
    preload_macau_graph()
    cameras = [StreamCamera(url, MACAU_STREETS[i] if i<len(MACAU_STREETS) else f"Cam {i+1}") for i, url in enumerate(streams)]
    states = [CameraState(name=cam.name, tracker=SimpleIOUTracker(), stream_url=cam.url) for cam in cameras]
    
    model = YOLO(MODEL_PATH)
    last_map_update_t = time.time()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    while True:
        now = time.time()
        
        if (now - last_map_update_t) > 5.0:
            threading.Thread(target=post_macau_traffic_data, args=(states, CLOUD_API_URL), daemon=True).start()
            log_jam_to_csv(states, CSV_LOG_FILE)
            last_map_update_t = now

        for idx, cam in enumerate(cameras):
            st = states[idx]
            if (now - st.last_proc_t) < PROC_INTERVAL_SEC: continue
            ret, frame = cam.get_frame()
            if not ret: continue
            st.last_proc_t = now

            ovr = CAM_OVERRIDES.get(st.name, {})
            if "crop" in ovr:
                x1, y1, x2, y2 = ovr["crop"]
                h_orig, w_orig = frame.shape[:2]
                y_s, y_e, x_s, x_e = max(0, y1), min(h_orig, y2), max(0, x1), min(w_orig, x2)
                if y_e > y_s and x_e > x_s: frame = frame[y_s:y_e, x_s:x_e]

            if frame is None or frame.size == 0: continue
            
            proc_size_override = ovr.get("proc_size", (PROC_W, PROC_H))
            if proc_size_override is None:
                proc = frame.copy()
            else:
                proc = cv2.resize(frame, proc_size_override)
                
            fh, fw = proc.shape[:2]

            # 🌟 [新增] 讀取動態比例尺與場地容量
            speed_scale = ovr.get("speed_scale", 1.0)
            capacity_scale = ovr.get("capacity_scale", 1.0)

            stat_spd_th = STATIONARY_SPEED_TH * speed_scale
            stop_spd_th = 3.0 * speed_scale
            norm_spd_th = 20.0 * speed_scale
            jam_occ_th = 0.40 # 畫面佔有率維持 40% 的滿載標準
            max_car_area = (fw * fh) * 0.10 

            # 1. 基礎 YOLO 辨識
            res = model.predict(proc, imgsz=ovr.get("imgsz", IMGSZ), conf=ovr.get("conf", CONF), classes=INCLUDE_CLASSES, device=DEVICE, verbose=False)[0]
            dets = [{"bbox": b.xyxy[0].cpu().numpy(), "conf": float(b.conf[0]), "cls_id": int(b.cls[0])} for b in res.boxes] if res.boxes else []

            # 2. 純動態捕捉 (信任模式)
            if st.use_motion and st.bg_subtractor is not None:
                gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                fg_mask = st.bg_subtractor.apply(gray)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
                
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # 🌟 微調下限到 50，避免把水波紋或樹葉微動當作車輛
                    if 50 < area < 8000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        m_bbox = np.array([x, y, x+w, y+h])
                        
                        overlap = False
                        for d in dets:
                            if iou_xyxy(m_bbox, d["bbox"]) > 0.1:
                                overlap = True
                                break
                        
                        if not overlap:
                            dets.append({"bbox": m_bbox, "conf": 0.5, "cls_id": 5})

            # 3. 送進 Tracker 追蹤
            tracks = st.tracker.update(dets, now, fw, fh, stationary_th=stat_spd_th)

            stable = [t for t in tracks if t.hits >= TRACK_MIN_HITS]
            active_vehicles = [t for t in stable if t.stationary_time < PARKED_TIME_SEC]
            parked_vehicles = [t for t in stable if t.stationary_time >= PARKED_TIME_SEC]

            v_act = len(active_vehicles)
            st.veh_history.append(v_act)
            
            history_list = list(st.veh_history)[-50:]
            median_v = sorted(history_list)[len(history_list)//2] if len(history_list) > 10 else 0
            
            occ = sum(min((t.bbox[2]-t.bbox[0])*(t.bbox[3]-t.bbox[1]), max_car_area) for t in active_vehicles) / (fw*fh) if fw*fh>0 else 0
            spd = np.median([t.speed_ema_px_s for t in active_vehicles]) if active_vehicles else 0
            stop = sum(1 for t in active_vehicles if t.speed_ema_px_s < stop_spd_th) / v_act if v_act > 0 else 0
            
            # --- 流量跨線偵測 (VPM) ---
            line_y = int(fh * LINE_Y_RATIO)
            for tr in active_vehicles:
                if tr.prev_center and tr.center and not tr.counted:
                    py = tr.prev_center[1]
                    cy = tr.center[1]
                    if (py < line_y and cy >= line_y) or (py > line_y and cy <= line_y):
                        st.flow_timestamps.append(now) 
                        tr.counted = True

            while st.flow_timestamps and now - st.flow_timestamps[0] > 60.0:
                st.flow_timestamps.popleft()
            
            current_vpm = len(st.flow_timestamps)

            # 🌟 4. 新版塞車指數與降溫邏輯 (結合場地容量係數) 🌟
            if v_act == 0: 
                raw_jam = 0.0
                for _ in range(5): 
                    st.jam_history.append(0.0)
            else:
                raw_jam = 0.30 * min(1, occ/jam_occ_th) + 0.20 * (1 - min(1, spd/norm_spd_th)) + 0.50 * stop
                
                # 🌟 [核心修改] 判斷塞車門檻乘上 capacity_scale！
                if median_v <= (3 * capacity_scale): 
                    raw_jam *= 0.2  
                elif median_v >= (12 * capacity_scale): 
                    raw_jam = min(1.0, raw_jam * 1.2) 
                else:
                    raw_jam *= 0.7  

                st.jam_history.append(raw_jam)

            jam_avg = sum(st.jam_history) / len(st.jam_history) if st.jam_history else 0.0
            
            if median_v <= (1 * capacity_scale):
                jam_avg = 0.0
                st.jam_history.clear()

            st.last_metrics = {"veh_active": v_act, "jam_avg": jam_avg, "flow_total_vpm": current_vpm}

            # 5. 渲染 UI
            anno = proc.copy()
            
            cv2.line(anno, (0, line_y), (fw, line_y), (255, 0, 255), 2)
            
            for tr in active_vehicles: 
                cv2.rectangle(anno, (int(tr.bbox[0]), int(tr.bbox[1])), (int(tr.bbox[2]), int(tr.bbox[3])), (0,255,0), 2)
            for tr in parked_vehicles:
                cv2.rectangle(anno, (int(tr.bbox[0]), int(tr.bbox[1])), (int(tr.bbox[2]), int(tr.bbox[3])), (0,165,255), 2)
                
            anno = cv2.resize(anno, (TILE_W, TILE_H))
            
            bar_color = (40, 167, 69) if jam_avg < 0.4 else (7, 193, 255) if jam_avg < 0.7 else (53, 53, 220)
            cv2.rectangle(anno, (0, TILE_H-8), (int(TILE_W*jam_avg), TILE_H), bar_color, -1)
            anno = draw_chinese_text_bg(anno, f"{st.name} 車:{v_act} 流/分:{current_vpm}", 5, 5, font_size=14, bg=bar_color)
            st.last_tile = anno

        if SHOW_UI:
            grid = np.zeros((GRID_ROWS*TILE_H, GRID_COLS*TILE_W, 3), dtype=np.uint8)
            for i, st in enumerate(states):
                if st.last_tile is not None:
                    r, c = i//GRID_COLS, i%GRID_COLS
                    grid[r*TILE_H:(r+1)*TILE_H, c*TILE_W:(c+1)*TILE_W] = st.last_tile
            cv2.imshow(WINDOW_NAME, grid)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    for c in cameras: c.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()