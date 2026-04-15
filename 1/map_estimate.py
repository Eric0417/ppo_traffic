import os
import cv2
import csv
import time
import math
import threading
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from queue import Queue, Empty
from typing import Optional, Tuple
import copy

import numpy as np
import torch
from ultralytics import YOLO

# 處理中文顯示
from PIL import Image, ImageDraw, ImageFont

# 進階路網分析與地圖繪製
import networkx as nx
try:
    import folium
except Exception:
    folium = None

try:
    import osmnx as ox
    ox.settings.log_console = False
    ox.settings.use_cache = True
except ImportError:
    ox = None

try:
    import openpyxl  # optional for XLSX
except Exception:
    openpyxl = None


# =========================
# CONFIG (基礎設定)
# =========================
MAX_CAMERAS = 60

STREAMS_FILE = r"D:\school\bsd\flow_estimate\streams.txt"
MODEL_PATH = r"D:\school\bsd\flow_estimate\yolo12x.pt"

# 中文字型路徑
FONT_PATH = r"C:\Windows\Fonts\msjh.ttc"

# 只偵測車輛類（COCO）：2=car, 3=motorcycle, 5=bus, 7=truck
INCLUDE_CLASSES = [2, 3, 5, 7]

DEVICE = 0 if torch.cuda.is_available() else "cpu"
CONF = 0.25
IMGSZ = 640
HALF = True if torch.cuda.is_available() else False

PROC_FPS_PER_CAM = 10.0
PROC_INTERVAL_SEC = 1.0 / PROC_FPS_PER_CAM
BATCH_SIZE = 5  

# 預設推論用尺寸
PROC_W, PROC_H = 640, 360

# =========================
# 🦅 鷹眼模式 (針對特定鏡頭覆蓋參數)
# =========================
CAM_OVERRIDES = {
    "亞馬喇前地之圓形地方向": {
        # 【新增】局部裁剪 (x_min, y_min, x_max, y_max)
        # 請根據你畫面的實際狀況，填入包含圓環的座標範圍
        # 假設原始畫面是 1920x1080，我們只切出中間偏下的圓環部分：
        "crop": (400, 300, 1500, 900), 
        
        "proc_size": (1280, 720), 
        "imgsz": 1280,            
        "conf": 0.10              # 信心門檻再降到 0.10 試試看
    }
}

GRID_COLS, GRID_ROWS = 6, 10
TILE_W, TILE_H = 320, 180  

LINE_Y_RATIO = 0.55
FLOW_WINDOW_SEC = 60.0

IOU_MATCH_TH = 0.30
TRACK_MAX_AGE = 6          
TRACK_MIN_HITS = 2         

O_JAM_REF = 0.35           
V_STOP_PX_S = 6.0          
FREE_SPEED_WINDOW_SEC = 300.0  
V_FREE_DEFAULT = 80.0      

W_OCC, W_SPD, W_STOP = 0.55, 0.35, 0.10  
SMOOTH_WINDOW_SIZE = 300 

LOG_DIR = r"D:\school\bsd\flow_estimate\logs"
LOG_FORMAT = "xlsx"  
LOG_EVERY_SEC = 5.0

SHOW_UI = True
WINDOW_NAME = "Traffic 60-Cam (YOLOv12x) | q=quit"

# =========================
# 澳門 60 路街道名稱
# =========================
MACAU_STREETS = [
    "提督馬路與高士德大馬路交界（向高士德大馬路方向）", "巴坡沙大馬路與青洲大馬路交界", 
    "提督馬路與罅些喇海軍上將巷交界", "提督馬路與高士德大馬路交界", 
    "沙梨頭海邊街與爹美刁施拿地大馬路交界", "勞動節大馬路", "慕拉士大馬路", 
    "友誼橋大馬路(向氹仔方向)", "友誼大馬路(向馬揸度博士大馬路方向)", "馬揸度博士大馬路", 
    "馬揸度博士大馬路(近勞工局)", "馬場北大馬路與馬場東大馬路交界", 
    "友誼圓形地與友誼橋大馬路交界(向港珠澳大橋入口方向)", "美副將大馬路與俾利喇街交界", 
    "東北大馬路與黑沙環中街交界向友誼圓形地方向", "黑沙環新街", "馬交石斜坡與俾利喇街交界", 
    "A2橋澳門出口", "友誼大馬路", "馬六甲街停車場", "松山隧道羅理基方向", 
    "羅理基博士大馬路行車隧道往松山隧道方向", "宋玉生廣場", "捐血中心", 
    "孫逸仙大馬路與城市日大馬路交界", "孫逸仙大馬路(近終審法院前地)", 
    "西灣湖廣場(向孫逸仙大馬路)", "西灣湖廣場(向西灣湖景大馬路)", "高士德與俾利喇街交界", 
    "華士古停車場", "美副將大馬路與連勝馬路交界", "松山隧道高士德方向", 
    "美副將大馬路與士多鳥拜斯大馬路交界", "高偉樂街與荷蘭園大馬路交界", "水坑尾", 
    "沙梨頭海邊街", "亞馬喇前地之圓形地方向", "亞馬喇前地", "亞馬喇前地巴士站", 
    "南灣大馬路與區華利前地交界", "南灣大馬路與殷皇子大馬路(向八角亭方向)", 
    "南灣大馬路與殷皇子大馬路(向殷皇子大馬路方向)", "巴素打爾古街近栢港停車場出口(向火船頭街方向)", 
    "新馬路(近議事亭前地)向南灣大馬路方向", "殷皇子大馬路與約翰四世大馬路交界", 
    "殷皇子大馬路與葡京路交界", "水坑尾街與南灣大馬路交界路口", "沙梨頭海邊街與林茂巷交界", 
    "沙梨頭海邊街與魚鰓巷交界(向新馬路方向)", "爹美刁斯拿地大馬路近栢港停車場", 
    "爹美刁施拿地大馬路與魚鱗巷交界(向十六浦方向)", "新馬路與巴素打爾古街交界", 
    "新馬路與南灣大馬路交界", "比厘喇馬忌士街與貨倉巷交界(向十六浦方向)", 
    "比厘喇馬忌士街與馬博士巷交界", "河邊新街與比厘喇馬忌士街交界(向媽閣廟方向)", 
    "河邊新街與航海學校街交界(向媽閣方向)", "河邊新街與鹽巷交界（向媽閣方向）", 
    "火船頭街近11號碼頭（向巴素打爾古街方向)", "火船頭街（近11號碼頭）（向河邊新街方向）"
]

ORIGINAL_COORDS = {
    "提督馬路與高士德大馬路交界（向高士德大馬路方向）": (22.20533, 113.54436),
    "巴坡沙大馬路與青洲大馬路交界": (22.21027, 113.54734),
    "提督馬路與罅些喇海軍上將巷交界": (22.20401, 113.54255),
    "提督馬路與高士德大馬路交界": (22.20525, 113.54427),
    "沙梨頭海邊街與爹美刁施拿地大馬路交界": (22.20248, 113.53801),
    "勞動節大馬路": (22.21059, 113.55409),
    "慕拉士大馬路": (22.20665, 113.55256),
    "友誼橋大馬路(向氹仔方向)": (22.20866, 113.56081),
    "友誼大馬路(向馬揸度博士大馬路方向)": (22.20474, 113.56060),
    "馬揸度博士大馬路": (22.20563, 113.55812),
    "馬揸度博士大馬路(近勞工局)": (22.20573, 113.55884),
    "馬場北大馬路與馬場東大馬路交界": (22.21385, 113.55444),
    "友誼圓形地與友誼橋大馬路交界(向港珠澳大橋入口方向)": (22.21175, 113.55951),
    "美副將大馬路與俾利喇街交界": (22.20496, 113.54877),
    "東北大馬路與黑沙環中街交界向友誼圓形地方向": (22.20938, 113.55765),
    "黑沙環新街": (22.20613, 113.55738),
    "馬交石斜坡與俾利喇街交界": (22.20669, 113.55032),
    "A2橋澳門出口": (22.212000, 113.555000), 
    "友誼大馬路": (22.19077, 113.54870),
    "馬六甲街停車場": (22.19606, 113.55367),
    "松山隧道羅理基方向": (22.19674, 113.55192),
    "羅理基博士大馬路行車隧道往松山隧道方向": (22.19619, 113.55196),
    "宋玉生廣場": (22.19006, 113.55043),
    "捐血中心": (22.18946, 113.54977),
    "孫逸仙大馬路與城市日大馬路交界": (22.18585, 113.54943),
    "孫逸仙大馬路(近終審法院前地)": (22.18239, 113.54111),
    "西灣湖廣場(向孫逸仙大馬路)": (22.18146, 113.53839),
    "西灣湖廣場(向西灣湖景大馬路)": (22.18044, 113.53762),
    "高士德與俾利喇街交界": (22.20290, 113.54709),
    "華士古停車場": (22.19639, 113.54650),
    "美副將大馬路與連勝馬路交界": (22.20625, 113.54715),
    "松山隧道高士德方向": (22.19889, 113.55073),
    "美副將大馬路與士多鳥拜斯大馬路交界": (22.20191, 113.55247),
    "高偉樂街與荷蘭園大馬路交界": (22.19768, 113.54638),
    "水坑尾": (22.19405, 113.54369),
    "沙梨頭海邊街": (22.20314, 113.54033),
    "亞馬喇前地之圓形地方向": (22.18895, 113.54317),
    "亞馬喇前地": (22.18990, 113.54339),
    "亞馬喇前地巴士站": (22.18944, 113.54336),
    "南灣大馬路與區華利前地交界": (22.19105, 113.53927),
    "南灣大馬路與殷皇子大馬路(向八角亭方向)": (22.19224, 113.54096),
    "南灣大馬路與殷皇子大馬路(向殷皇子大馬路方向)": (22.19219, 113.54089),
    "巴素打爾古街近栢港停車場出口(向火船頭街方向)": (22.19765, 113.53669),
    "新馬路(近議事亭前地)向南灣大馬路方向": (22.19234, 113.54081),
    "殷皇子大馬路與約翰四世大馬路交界": (22.19111, 113.54202),
    "殷皇子大馬路與葡京路交界": (22.18993, 113.54341),
    "水坑尾街與南灣大馬路交界路口": (22.19273, 113.54340),
    "沙梨頭海邊街與林茂巷交界": (22.20250, 113.53806),
    "沙梨頭海邊街與魚鰓巷交界(向新馬路方向)": (22.20018, 113.53728),
    "爹美刁斯拿地大馬路近栢港停車場": (22.19829, 113.53657),
    "爹美刁施拿地大馬路與魚鱗巷交界(向十六浦方向)": (22.20174, 113.53727),
    "新馬路與巴素打爾古街交界": (22.19642, 113.53648),
    "新馬路與南灣大馬路交界": (22.19225, 113.54085),
    "比厘喇馬忌士街與貨倉巷交界(向十六浦方向)": (22.19108, 113.53459),
    "比厘喇馬忌士街與馬博士巷交界": (22.19218, 113.53400),
    "河邊新街與比厘喇馬忌士街交界(向媽閣廟方向)": (22.18947, 113.53268),
    "河邊新街與航海學校街交界(向媽閣方向)": (22.18746, 113.53120),
    "河邊新街與鹽巷交界（向媽閣方向）": (22.19011, 113.53309),
    "火船頭街近11號碼頭（向巴素打爾古街方向)": (22.19630, 113.53647),
    "火船頭街（近11號碼頭）（向河邊新街方向）": (22.19319, 113.53520)
}

# 防重疊推開演算法
def resolve_overlaps(coords_dict, min_dist=0.00015):
    coords = {k: list(v) for k, v in coords_dict.items()}
    keys = list(coords.keys())
    for _ in range(15): 
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                k1, k2 = keys[i], keys[j]
                lat1, lon1 = coords[k1]
                lat2, lon2 = coords[k2]
                dist = math.hypot(lat1 - lat2, lon1 - lon2)
                if dist < min_dist:
                    angle = math.atan2(lat1 - lat2, lon1 - lon2) if dist != 0 else i * 0.5
                    push = (min_dist - (dist if dist != 0 else 0.00001)) / 2.0
                    coords[k1][0] += push * math.sin(angle)
                    coords[k1][1] += push * math.cos(angle)
                    coords[k2][0] -= push * math.sin(angle)
                    coords[k2][1] -= push * math.cos(angle)
    return {k: tuple(v) for k, v in coords.items()}

STREET_COORDS = resolve_overlaps(ORIGINAL_COORDS)

# =========================
# OSMnx 澳門全境路網圖擴散 (Graph Diffusion)
# =========================
MACAU_GRAPH = None

def preload_macau_graph():
    """下載全澳門半島真實路網"""
    global MACAU_GRAPH
    if ox is None:
        print("未安裝 osmnx，無法使用圖擴散渲染。將使用備用地圖生成模式。")
        return
    
    print("正在下載澳門半島真實路網 (首次執行需約 1 分鐘，後續將自動快取)...")
    try:
        # 相容 osmnx 舊版與新版的參數設定
        try:
            MACAU_GRAPH = ox.graph_from_bbox(22.2170, 22.1780, 113.5650, 113.5250, network_type='drive')
        except TypeError:
            MACAU_GRAPH = ox.graph_from_bbox(bbox=(113.5250, 22.1780, 113.5650, 22.2170), network_type='drive')
        print(f"✅ 澳門路網載入成功！共解析出 {len(MACAU_GRAPH.nodes)} 個路口。")
    except Exception as e:
        print(f"❌ 路網下載失敗: {e}")

def run_heat_diffusion(G, source_scores, iterations=50):
    """將60路鏡頭真實數據沿著街道擴散至全澳門"""
    scores = {n: None for n in G.nodes()}
    for n, s in source_scores.items():
        scores[n] = s

    for _ in range(iterations):
        new_scores = scores.copy()
        for n in G.nodes():
            if n in source_scores:
                continue 
            neighbors = list(G.successors(n)) + list(G.predecessors(n))
            valid_scores = [scores[nb] for nb in neighbors if scores[nb] is not None]
            if valid_scores:
                new_scores[n] = sum(valid_scores) / len(valid_scores)
        scores = new_scores
        
    for n in G.nodes():
        if scores[n] is None:
            scores[n] = 0.0
    return scores

def update_macau_map_diffusion(states, filepath="macau_traffic.html"):
    if folium is None or MACAU_GRAPH is None: return

    current_scores = {st.name: (st.last_metrics.get("jam_avg", 0.0) if st.last_metrics else 0.0) for st in states}

    def get_color(score):
        if score < 0.4: return "#28a745"   # 綠 (暢通)
        elif score < 0.7: return "#ffc107" # 黃 (車多)
        else: return "#dc3545"             # 紅 (壅塞)

    # 1. 將實體鏡頭映射到圖節點
    camera_node_scores = {}
    for name, coord in STREET_COORDS.items():
        if name in current_scores:
            lat, lon = coord
            nearest_node = ox.distance.nearest_nodes(MACAU_GRAPH, X=lon, Y=lat)
            camera_node_scores[nearest_node] = current_scores[name]

    # 2. 執行熱擴散演算
    node_scores = run_heat_diffusion(MACAU_GRAPH, camera_node_scores, iterations=50)

    # 3. 初始化深色底圖
    m = folium.Map(
        location=[22.1965, 113.5450], 
        zoom_start=15, min_zoom=14, max_zoom=18,
        min_lat=22.1750, max_lat=22.2200, min_lon=113.5250, max_lon=113.5650,
        max_bounds=True,
        tiles="CartoDB dark_matter"
    )

    # 4. 畫出所有受影響的街道
    for u, v, key, data in MACAU_GRAPH.edges(keys=True, data=True):
        edge_score = (node_scores[u] + node_scores[v]) / 2.0
        color = get_color(edge_score)
        
        if 'geometry' in data:
            coords = [(lat, lon) for lon, lat in data['geometry'].coords]
        else:
            coords = [(MACAU_GRAPH.nodes[u]['y'], MACAU_GRAPH.nodes[u]['x']), 
                      (MACAU_GRAPH.nodes[v]['y'], MACAU_GRAPH.nodes[v]['x'])]
        
        # 過濾掉極短的不重要線段提升效能
        if data.get('length', 0) > 10:
            folium.PolyLine(locations=coords, color=color, weight=2.5, opacity=0.7).add_to(m)

    # 5. 繪製實體鏡頭節點 (發光點)
    for st in states:
        if st.name not in STREET_COORDS: continue
        lat, lon = STREET_COORDS[st.name]
        score = current_scores.get(st.name, 0.0)
        marker_color = get_color(score)
        
        veh_active = st.last_metrics.get("veh_active", 0) if st.last_metrics else 0
        flow_total = st.last_metrics.get("flow_total_vpm", 0.0) if st.last_metrics else 0.0
            
        popup_html = f"""
        <div style="font-family: Arial, 'Microsoft JhengHei', sans-serif; width: 220px;">
            <h4 style="margin-bottom: 5px; color: #333;">{st.name}</h4>
            <hr style="margin: 5px 0;">
            <b>塞車指數:</b> {score:.2f} <span style="font-size:10px; color:gray;">(實體鏡頭數據)</span><br>
            <b>當下車流:</b> {flow_total:.1f} 輛/分鐘<br>
            <b>畫面車數:</b> {veh_active} 輛
        </div>
        """
        folium.CircleMarker(
            location=[lat, lon], radius=5, 
            popup=folium.Popup(popup_html, max_width=250),
            color="#ffffff", weight=1.5, fill=True, fill_color=marker_color, fill_opacity=1.0
        ).add_to(m)

    try: m.save(filepath)
    except Exception: pass


# =========================
# Utils
# =========================
def load_streams(path: str, max_n: int) -> list[str]:
    if not os.path.exists(path): return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            out.append(s)
            if len(out) >= max_n: break
    return out

def clamp_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    x1 = float(max(0, min(w - 1, xyxy[0])))
    y1 = float(max(0, min(h - 1, xyxy[1])))
    x2 = float(max(0, min(w - 1, xyxy[2])))
    y2 = float(max(0, min(h - 1, xyxy[3])))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    union = (max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)) + (max(0.0, bx2 - bx1) * max(0.0, by2 - by1)) - inter
    return float(inter / union) if union > 1e-6 else 0.0

def color_for_id(tid: int) -> tuple[int, int, int]:
    return int((tid * 97) % 255), int((tid * 17) % 255), int((tid * 37) % 255)

def draw_chinese_text_bg(img, text: str, x: int, y: int, font_size=16, fg=(255, 255, 255), bg=(0, 0, 0)):
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        font = ImageFont.load_default()
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0]-4, bbox[1]-4, bbox[2]+4, bbox[3]+4], fill=bg)
    draw.text((x, y), text, font=font, fill=fg)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =========================
# Capture
# =========================
class StreamCamera:
    def __init__(self, url: str, name: str, reopen_delay_sec: float = 1.0, read_sleep_sec: float = 0.01):
        self.url, self.name, self.reopen_delay_sec, self.read_sleep_sec = url, name, reopen_delay_sec, read_sleep_sec
        self.cap = self._open_capture()
        self.ret, self.frame = False, None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open_capture(self):
        try: cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        except Exception: cap = cv2.VideoCapture(self.url)
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        return cap

    def _reopen(self):
        try: self.cap.release()
        except Exception: pass
        self.cap = self._open_capture()
        time.sleep(self.reopen_delay_sec)

    def _update(self):
        while self.running:
            if not self.cap.isOpened():
                self._reopen(); continue
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = bool(ret), (frame if ret else None)
            if not ret: self._reopen()
            else: time.sleep(self.read_sleep_sec)

    def get_frame(self):
        with self.lock:
            if not self.ret or self.frame is None: return False, None
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        try: self.thread.join(timeout=2.0)
        except Exception: pass
        try: self.cap.release()
        except Exception: pass

# =========================
# Simple IoU Tracker
# =========================
@dataclass
class Track:
    tid: int
    bbox: np.ndarray
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
        self.iou_th, self.max_age, self._next_id = float(iou_th), int(max_age), 1
        self.tracks: list[Track] = []

    @staticmethod
    def _center(b: np.ndarray) -> tuple[float, float]:
        return (float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0))

    def update(self, detections: list[dict], t: float, frame_w: int, frame_h: int) -> list[Track]:
        for tr in self.tracks: tr.time_since_update += 1

        matches = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                iou = iou_xyxy(tr.bbox, det["bbox"])
                if iou >= self.iou_th: matches.append((iou, ti, di))

        matches.sort(key=lambda x: x[0], reverse=True)
        used_tracks, used_dets = set(), set()

        for iou, ti, di in matches:
            if ti in used_tracks or di in used_dets: continue
            used_tracks.add(ti); used_dets.add(di)

            tr, det = self.tracks[ti], detections[di]
            new_bbox = clamp_xyxy(det["bbox"], frame_w, frame_h)

            tr.prev_center = tr.center if tr.center is not None else self._center(tr.bbox)
            tr.bbox, tr.conf, tr.cls_id, tr.center = new_bbox, float(det["conf"]), int(det["cls_id"]), self._center(new_bbox)

            dt = max(1e-3, float(t - tr.last_t)) if tr.last_t > 0 else None
            if dt is not None:
                dx, dy = tr.center[0] - tr.prev_center[0], tr.center[1] - tr.prev_center[1]
                tr.speed_px_s = float(math.sqrt(dx * dx + dy * dy) / dt)
                tr.speed_ema_px_s = 0.6 * tr.speed_ema_px_s + 0.4 * tr.speed_px_s
            tr.last_t, tr.hits, tr.time_since_update = float(t), tr.hits + 1, 0

        for di, det in enumerate(detections):
            if di in used_dets: continue
            bbox = clamp_xyxy(det["bbox"], frame_w, frame_h)
            self.tracks.append(Track(
                tid=self._next_id, bbox=bbox, conf=float(det["conf"]), cls_id=int(det["cls_id"]),
                hits=1, time_since_update=0, last_t=float(t), prev_center=None, center=self._center(bbox),
                speed_px_s=0.0, speed_ema_px_s=0.0, crossed=False,
            ))
            self._next_id += 1

        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]
        return [tr for tr in self.tracks if tr.time_since_update == 0]

# =========================
# Metrics
# =========================
@dataclass
class CameraState:
    name: str
    tracker: SimpleIOUTracker
    cross_up_times: deque = None
    cross_down_times: deque = None
    cross_up_total: int = 0
    cross_down_total: int = 0
    speed_median_hist: deque = None
    jam_history: deque = None       
    last_tile: Optional[np.ndarray] = None
    last_proc_t: float = 0.0
    last_log_t: float = 0.0
    last_metrics: dict = None

    def __post_init__(self):
        if self.cross_up_times is None: self.cross_up_times = deque()
        if self.cross_down_times is None: self.cross_down_times = deque()
        if self.speed_median_hist is None: self.speed_median_hist = deque()
        if self.jam_history is None: self.jam_history = deque(maxlen=SMOOTH_WINDOW_SIZE)
        if self.last_metrics is None: self.last_metrics = {}

def estimate_v_free(state: CameraState, now_t: float) -> float:
    while state.speed_median_hist and state.speed_median_hist[0][0] < now_t - FREE_SPEED_WINDOW_SEC:
        state.speed_median_hist.popleft()
    if len(state.speed_median_hist) < 20: return float(V_FREE_DEFAULT)
    speeds = np.array([s for _, s in state.speed_median_hist], dtype=np.float32)
    return max(float(V_FREE_DEFAULT), float(np.percentile(speeds, 90)))

def update_crossings(state: CameraState, tracks: list[Track], line_y_ratio: float, frame_h: int, now_t: float):
    line_y = int(frame_h * line_y_ratio)
    while state.cross_up_times and state.cross_up_times[0] < now_t - FLOW_WINDOW_SEC: state.cross_up_times.popleft()
    while state.cross_down_times and state.cross_down_times[0] < now_t - FLOW_WINDOW_SEC: state.cross_down_times.popleft()

    for tr in tracks:
        if tr.hits < TRACK_MIN_HITS or tr.crossed or tr.prev_center is None or tr.center is None: continue
        y0, y1 = tr.prev_center[1], tr.center[1]
        if y0 < line_y <= y1:
            state.cross_down_total += 1
            state.cross_down_times.append(now_t); tr.crossed = True
        elif y0 > line_y >= y1:
            state.cross_up_total += 1
            state.cross_up_times.append(now_t); tr.crossed = True

def compute_metrics(state: CameraState, tracks: list[Track], now_t: float, frame_w: int, frame_h: int) -> dict:
    roi_area = float(frame_w * frame_h)
    stable = [tr for tr in tracks if tr.hits >= TRACK_MIN_HITS]
    veh_active, raw = len(stable), 0.0 
    dynamic_v_stop = V_STOP_PX_S * (frame_h / 360.0)

    if veh_active > 0:
        area_sum, speeds, stop_cnt = 0.0, [], 0
        for tr in stable:
            x1, y1, x2, y2 = tr.bbox
            area_sum += max(0.0, x2 - x1) * max(0.0, y2 - y1)
            spd = float(tr.speed_ema_px_s)
            speeds.append(spd)
            if spd < dynamic_v_stop: stop_cnt += 1

        occupancy = float(max(0.0, min(1.0, area_sum / roi_area)))
        median_speed = float(np.median(np.array(speeds, dtype=np.float32))) if speeds else 0.0
        stop_ratio = float(stop_cnt / max(1, veh_active))

        if median_speed > 1e-3: state.speed_median_hist.append((now_t, median_speed))
        v_free = float(estimate_v_free(state, now_t))

        o_norm = float(max(0.0, min(1.0, occupancy / max(1e-6, O_JAM_REF))))
        s_norm = 1.0 - float(max(0.0, min(1.0, median_speed / max(1e-6, v_free))))
        raw = float(max(0.0, min(1.0, W_OCC * o_norm + W_SPD * s_norm + W_STOP * stop_ratio)))

    state.jam_history.append(raw)
    jam_avg = sum(state.jam_history) / len(state.jam_history) if len(state.jam_history) > 0 else 0.0

    return dict(veh_active=int(veh_active), occupancy=float(occupancy if veh_active else 0.0), 
                median_speed=float(median_speed if veh_active else 0.0), v_free=float(v_free if veh_active else 80.0), 
                stop_ratio=float(stop_ratio if veh_active else 0.0), jam_raw=float(raw), jam_avg=float(jam_avg))

def flow_vpm_from_deque(dq: deque) -> float:
    return float(len(dq) * 60.0 / FLOW_WINDOW_SEC) if FLOW_WINDOW_SEC > 1e-6 else 0.0

# =========================
# Logging
# =========================
LOG_HEADER = [
    "ts_iso", "cam_idx", "cam_name", "veh_active", "flow_total_vpm", "flow_up_vpm", "flow_down_vpm",
    "cross_total", "cross_up_total", "cross_down_total", "occupancy", "median_speed_px_s",
    "v_free_px_s", "stop_ratio", "jam_raw", "jam_avg", "proc_fps_est",
]

class LogWorker(threading.Thread):
    def __init__(self, out_dir: str, fmt: str, queue: Queue):
        super().__init__(daemon=True)
        self.out_dir, self.fmt, self.q = out_dir, fmt.lower(), queue
        self.running, self.current_date = True, None
        self.csv_f = self.csv_writer = self.xlsx_path = self.xlsx_wb = self.xlsx_ws = None
        self.xlsx_rows_since_save, self.last_xlsx_save_t = 0, time.time()
        os.makedirs(self.out_dir, exist_ok=True)

    def _rotate(self, date_str: str):
        if self.csv_f is not None:
            try: self.csv_f.flush(); self.csv_f.close()
            except Exception: pass
        self.csv_f, self.csv_writer = None, None
        if self.xlsx_wb is not None:
            try: self.xlsx_wb.save(self.xlsx_path)
            except Exception: pass
        self.xlsx_wb = self.xlsx_ws = self.xlsx_path = None
        self.xlsx_rows_since_save, self.current_date = 0, date_str

        if self.fmt in ("csv", "both"):
            csv_path = os.path.join(self.out_dir, f"traffic_{date_str}.csv")
            is_new = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
            self.csv_f = open(csv_path, "a", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.csv_f, fieldnames=LOG_HEADER)
            if is_new: self.csv_writer.writeheader(); self.csv_f.flush()

        if self.fmt in ("xlsx", "both"):
            if openpyxl is None: raise RuntimeError("請 pip install openpyxl")
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
        date_str = row["ts_iso"][:10]
        if self.current_date != date_str: self._rotate(date_str)
        if self.csv_writer is not None: self.csv_writer.writerow(row)
        if self.xlsx_ws is not None:
            self.xlsx_ws.append([row.get(k, "") for k in LOG_HEADER])
            self.xlsx_rows_since_save += 1
            now = time.time()
            if self.xlsx_rows_since_save >= 200 or (now - self.last_xlsx_save_t) > 60:
                self.xlsx_wb.save(self.xlsx_path)
                self.xlsx_rows_since_save, self.last_xlsx_save_t = 0, now
        if self.csv_f is not None: self.csv_f.flush()

    def run(self):
        while self.running or not self.q.empty():
            try: row = self.q.get(timeout=0.5)
            except Empty: continue
            try: self._write_row(row)
            except Exception: pass
        try:
            if self.csv_f is not None: self.csv_f.flush(); self.csv_f.close()
            if self.xlsx_wb is not None and self.xlsx_path is not None: self.xlsx_wb.save(self.xlsx_path)
        except Exception: pass

    def stop(self): self.running = False

# =========================
# Drawing
# =========================
def annotate(frame: np.ndarray, cam_name: str, tracks: list[Track], metrics: dict, line_y_ratio: float, flow_up_vpm: float, flow_down_vpm: float) -> np.ndarray:
    img = frame.copy()
    line_y = int(img.shape[0] * line_y_ratio)
    cv2.line(img, (0, line_y), (img.shape[1], line_y), (255, 0, 0), 2)

    for tr in tracks:
        if tr.hits < TRACK_MIN_HITS: continue
        x1, y1, x2, y2 = tr.bbox.astype(int)
        col = color_for_id(tr.tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"id:{tr.tid} v:{tr.speed_ema_px_s:.0f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    jam = float(metrics.get("jam_avg", 0.0))
    veh = int(metrics.get("veh_active", 0))
    flow_total = flow_up_vpm + flow_down_vpm
    jam_col = (0, int(255 * (1 - jam)), int(255 * jam))

    img = draw_chinese_text_bg(img, f"{cam_name}", 8, 10, font_size=16, fg=(255, 255, 255), bg=(30, 30, 30))
    img = draw_chinese_text_bg(img, f"車輛:{veh} | 車流:{flow_total:.1f}/分 | 塞車指數:{jam:.2f}", 8, 38, font_size=14, fg=(255, 255, 255), bg=jam_col)

    bar_w = int((img.shape[1] - 16) * jam)
    cv2.rectangle(img, (8, img.shape[0] - 14), (img.shape[1] - 8, img.shape[0] - 6), (50, 50, 50), -1)
    cv2.rectangle(img, (8, img.shape[0] - 14), (8 + bar_w, img.shape[0] - 6), jam_col, -1)

    return img

# =========================
# Main
# =========================
def main():
    streams = load_streams(STREAMS_FILE, MAX_CAMERAS)
    if len(streams) == 0: raise FileNotFoundError(f"請建立 {STREAMS_FILE}。")

    print(f"Loaded {len(streams)} streams.")
    os.makedirs(LOG_DIR, exist_ok=True)

    # 程式啟動時先下載與準備全澳門的 Graph 幾何形狀
    preload_macau_graph()

    cameras = []
    for i, url in enumerate(streams):
        name = MACAU_STREETS[i] if i < len(MACAU_STREETS) else f"Cam {i+1:02d}"
        cameras.append(StreamCamera(url, name))

    states = []
    for i in range(len(cameras)):
        name = MACAU_STREETS[i] if i < len(MACAU_STREETS) else f"Cam {i+1:02d}"
        states.append(CameraState(name=name, tracker=SimpleIOUTracker(iou_th=IOU_MATCH_TH, max_age=TRACK_MAX_AGE)))

    print(f"Loading YOLO model: {MODEL_PATH} on device={DEVICE}")
    model = YOLO(MODEL_PATH)
    try: model.fuse()
    except Exception: pass

    log_q = Queue(maxsize=20000)
    worker = LogWorker(LOG_DIR, LOG_FORMAT, log_q)
    worker.start()

    blank_tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    
    last_map_update_t = time.time()
    map_opened = False 

    try:
        if SHOW_UI: cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            now = time.time()

            # 地圖生成與自動開啟 (每 5 秒覆蓋產生一次)
            if folium is not None and (now - last_map_update_t) > 5.0:
                map_path = os.path.abspath("macau_traffic.html")
                # 使用熱傳導擴散渲染地圖
                update_macau_map_diffusion(states, filepath=map_path)
                last_map_update_t = now
                
                # 自動用瀏覽器彈出網頁 (僅第一次)
                if not map_opened and os.path.exists(map_path):
                    webbrowser.open('file://' + map_path)
                    map_opened = True

            batch_list = []

            for idx in range(len(cameras)):
                st = states[idx]
                if st.last_proc_t > 0 and (now - st.last_proc_t) < PROC_INTERVAL_SEC: continue
                ret, frame = cameras[idx].get_frame()
                if not ret or frame is None:
                    if st.last_tile is None:
                        st.last_tile = draw_chinese_text_bg(blank_tile.copy(), f"{st.name} 載入中...", 10, TILE_H // 2, font_size=12, fg=(255, 255, 255), bg=(0, 0, 255))
                    continue

                # 抓取該鏡頭的專屬「鷹眼覆蓋參數」
                ovr = CAM_OVERRIDES.get(st.name, {})
                pw, ph = ovr.get("proc_size", (PROC_W, PROC_H))

                proc = cv2.resize(frame, (pw, ph))
                batch_list.append((idx, proc, ovr))
                st.last_proc_t = now

            # 將畫面根據 imgsz 和 conf 動態分組批次推理
            batches = {}
            for idx, proc, ovr in batch_list:
                imgsz = ovr.get("imgsz", IMGSZ)
                conf = ovr.get("conf", CONF)
                key = (imgsz, conf)
                if key not in batches:
                    batches[key] = {'indices': [], 'frames': []}
                batches[key]['indices'].append(idx)
                batches[key]['frames'].append(proc)

            for (b_imgsz, b_conf), group in batches.items():
                frames = group['frames']
                indices = group['indices']

                for b0 in range(0, len(frames), BATCH_SIZE):
                    sub_frames = frames[b0:b0 + BATCH_SIZE]
                    sub_indices = indices[b0:b0 + BATCH_SIZE]

                    results = model.predict(sub_frames, imgsz=b_imgsz, conf=b_conf, classes=INCLUDE_CLASSES, device=DEVICE, half=HALF, verbose=False)

                    for res, cam_idx, proc_frame in zip(results, sub_indices, sub_frames):
                        st = states[cam_idx]
                        t_cam = time.time()
                        fh, fw = proc_frame.shape[:2]

                        dets = []
                        if res.boxes is not None and len(res.boxes) > 0:
                            xyxy, confs, clss = res.boxes.xyxy.detach().cpu().numpy(), res.boxes.conf.detach().cpu().numpy(), res.boxes.cls.detach().cpu().numpy().astype(int)
                            for i in range(len(xyxy)):
                                dets.append({"bbox": xyxy[i].astype(np.float32), "conf": float(confs[i]), "cls_id": int(clss[i])})

                        tracks = st.tracker.update(dets, t_cam, fw, fh)
                        update_crossings(st, tracks, line_y_ratio=LINE_Y_RATIO, frame_h=fh, now_t=t_cam)
                        flow_up_vpm, flow_down_vpm = flow_vpm_from_deque(st.cross_up_times), flow_vpm_from_deque(st.cross_down_times)

                        metrics = compute_metrics(st, tracks, now_t=t_cam, frame_w=fw, frame_h=fh)
                        metrics["flow_total_vpm"] = flow_up_vpm + flow_down_vpm
                        st.last_metrics = metrics

                        st.last_tile = cv2.resize(annotate(proc_frame, st.name, tracks, metrics, LINE_Y_RATIO, flow_up_vpm, flow_down_vpm), (TILE_W, TILE_H))

                        if (t_cam - st.last_log_t) >= LOG_EVERY_SEC:
                            st.last_log_t = t_cam
                            ts_iso = datetime.fromtimestamp(t_cam).isoformat(timespec="seconds")
                            row = {"ts_iso": ts_iso, "cam_idx": int(cam_idx + 1), "cam_name": st.name, "veh_active": int(metrics["veh_active"]), "flow_total_vpm": float(metrics["flow_total_vpm"]), "flow_up_vpm": float(flow_up_vpm), "flow_down_vpm": float(flow_down_vpm), "cross_total": int(st.cross_up_total + st.cross_down_total), "cross_up_total": int(st.cross_up_total), "cross_down_total": int(st.cross_down_total), "occupancy": float(metrics["occupancy"]), "median_speed_px_s": float(metrics["median_speed"]), "v_free_px_s": float(metrics["v_free"]), "stop_ratio": float(metrics["stop_ratio"]), "jam_raw": float(metrics["jam_raw"]), "jam_avg": float(metrics["jam_avg"]), "proc_fps_est": float(PROC_FPS_PER_CAM)}
                            try: log_q.put_nowait(row)
                            except Exception: pass

            if SHOW_UI:
                grid = np.zeros((GRID_ROWS * TILE_H, GRID_COLS * TILE_W, 3), dtype=np.uint8)
                for i in range(GRID_ROWS * GRID_COLS):
                    r, c = i // GRID_COLS, i % GRID_COLS
                    y0, y1, x0, x1 = r * TILE_H, (r + 1) * TILE_H, c * TILE_W, (c + 1) * TILE_W
                    if i < len(states) and states[i].last_tile is not None: grid[y0:y1, x0:x1] = states[i].last_tile
                    else: grid[y0:y1, x0:x1] = blank_tile

                cv2.imshow(WINDOW_NAME, grid)
                if cv2.waitKey(1) & 0xFF == ord("q"): break

            time.sleep(0.005)

    finally:
        print("Stopping...")
        for cam in cameras: cam.stop()
        worker.stop(); worker.join(timeout=3.0)
        if SHOW_UI: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()