#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ============================================================================
# v2_detector.py — Macau Traffic Real-Time Detector v2 (111 Cameras)
# ============================================================================
# 基於 main/map_estimate.py 擴充，支援全部 111 支澳門即時交通攝影機
# 包含七大類型識別: 半島道路 / 氹仔路環 / 跨海大橋 / 隧道 / 口岸 /
#                    港珠澳連接路 / 人工島通關車道
#
# 前綴對應:
#   m = 澳門半島一般道路  t = 氹仔/路環/隧道  b = 跨海大橋
#   a = 連接道路  p = 港珠澳邊檢  z = 人工島通關  h = 橫琴口岸
# ============================================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ⚠️ 必須在 import cv2 之前設定，否則 FFmpeg 逾時無效
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "timeout;8000000;stimeout;8000000;rw_timeout;8000000"
)

import cv2
import time
import math
import threading
import json
import requests
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from queue import Queue, Empty
from typing import Optional, Tuple
import copy

import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from common.camera import StreamCamera
from common.tracker import iou_xyxy

try:
    import osmnx as ox
    ox.settings.log_console = False
    ox.settings.use_cache = True
except ImportError:
    ox = None

# =========================
# CONFIG — v2 111鏡頭設定
# =========================
MAX_CAMERAS = 111
CLOUD_API_URL = "https://macau-traffic.onrender.com/api/update"
LOCAL_API_URL = "http://localhost:8000/api/update"  # 本地網站

STREAMS_FILE = r"D:\school\bsd\1\m3u8\macau_streams.txt"
MODEL_PATH = r"D:\school\bsd\1\yolo12x.pt"
FONT_PATH = r"C:\Windows\Fonts\msjh.ttc"

INCLUDE_CLASSES = [2, 3, 5, 7]

DEVICE = 0 if torch.cuda.is_available() else "cpu"
HALF = True if torch.cuda.is_available() else False
CONF = 0.25
IMGSZ = 416
PROC_FPS_PER_CAM = 5.0
PROC_INTERVAL_SEC = 1.0 / PROC_FPS_PER_CAM
PROC_W, PROC_H = 416, 240
UI_FPS = 10.0
PERF_LOG_INTERVAL = 30.0

# 違停過濾
PARKED_TIME_SEC = 120.0
STATIONARY_SPEED_TH = 2.0

# 動態攝影機覆蓋設定 (針對特殊鏡頭)
CAM_OVERRIDES = {
    # === 澳門半島廣角/圓形地 ===
    "亞馬喇前地之圓形地方向": {
        "proc_size": None, "imgsz": 1280, "conf": 0.15,
        "speed_scale": 0.2, "capacity_scale": 4.0,
    },
    "亞馬喇前地巴士站": {
        "proc_size": None, "imgsz": 1280, "conf": 0.15,
        "speed_scale": 0.3, "capacity_scale": 3.0,
    },
    # === 跨海大橋 (遠景, 車輛較小, 降低速度標準) ===
    "友誼大橋近港澳碼頭出口交匯處": {"speed_scale": 0.5},
    "友誼大橋近澳門端之橋峰望澳門方向": {"speed_scale": 0.4, "conf": 0.15},
    "友誼大橋近氹仔端之橋峰望氹仔方向": {"speed_scale": 0.4, "conf": 0.15},
    "友誼大橋近北安入口交匯處": {"speed_scale": 0.5},
    "友誼大橋近氹仔端出口": {"speed_scale": 0.5},
    "西灣大橋澳門方向": {"speed_scale": 0.45, "conf": 0.15},
    "西灣大橋氹仔方向": {"speed_scale": 0.45, "conf": 0.15},
    "西灣大橋中段望澳門方向": {"speed_scale": 0.35, "conf": 0.10},
    "澳門大橋近澳門端入口": {"speed_scale": 0.45, "conf": 0.15},
    "澳門大橋近澳門端(向氹仔方向)": {"speed_scale": 0.45, "conf": 0.15},
    "澳門大橋中段澳門往氹仔方向": {"speed_scale": 0.35, "conf": 0.10},
    "澳門大橋中段氹仔往澳門方向": {"speed_scale": 0.35, "conf": 0.10},
    "澳門大橋近氹仔端出入口": {"speed_scale": 0.45, "conf": 0.15},
    "澳門大橋近人工島澳門口岸交匯處": {"speed_scale": 0.45, "conf": 0.15},
    # === 港珠澳口岸 (邊檢車道, 車輛慢速) ===
    "港珠澳邊檢北大馬路": {"speed_scale": 0.6},
    "港珠澳邊檢南大馬路": {"speed_scale": 0.6},
    "港珠澳邊檢南大馬路 (入境車道)": {"speed_scale": 0.5},
    "港珠澳人工島往珠海通關車道A": {"speed_scale": 0.4, "capacity_scale": 2.0},
    "港珠澳人工島往澳門通關車道A": {"speed_scale": 0.4, "capacity_scale": 2.0},
    "港珠澳人工島往澳門通關車道B": {"speed_scale": 0.4, "capacity_scale": 2.0},
    "港珠澳人工島往珠海通關車道B": {"speed_scale": 0.4, "capacity_scale": 2.0},
    # === 橫琴口岸 ===
    "橫琴口岸澳門口岸區平台層 A": {"speed_scale": 0.5, "conf": 0.15},
    "橫琴口岸澳門口岸區平台層 B": {"speed_scale": 0.5, "conf": 0.15},
    "橫琴口岸澳門口岸區平台層 C": {"speed_scale": 0.5, "conf": 0.15},
    "橫琴口岸澳門口岸區平台層 D": {"speed_scale": 0.5, "conf": 0.15},
    # === 隧道 ===
    "松山隧道高士德方向": {"speed_scale": 0.7, "conf": 0.2},
    "松山隧道羅理基方向": {"speed_scale": 0.7, "conf": 0.2},
    "羅理基博士大馬路行車隧道往松山隧道方向": {"speed_scale": 0.7},
    "西灣大橋下層隧道(向澳門方向)": {"speed_scale": 0.5, "conf": 0.15},
    "路氹連貫公路下層隧道": {"speed_scale": 0.7, "conf": 0.2},
}

# v2 大網格: 10x12=120格 (111鏡頭 + 9備用)
GRID_COLS, GRID_ROWS = 10, 12
TILE_W, TILE_H = 320, 180
LINE_Y_RATIO = 0.55
SMOOTH_WINDOW_SIZE = 300
TRACK_MIN_HITS = 1
SHOW_UI = True
WINDOW_NAME = "Macau Traffic 111-Cam v2 | q=quit"


# MACAU_STREETS in annotated file order (geographic)
# 111 entries

MACAU_STREETS = [
    '新馬路與南灣大馬路交界',
    '水坑尾街與南灣大馬路交界路口',
    '新馬路與巴素打爾古街交界',
    '新馬路(近議事亭前地)向南灣大馬路方向',
    '南灣大馬路與區華利前地交界',
    '南灣大馬路與殷皇子大馬路(向八角亭方向)',
    '南灣大馬路與殷皇子大馬路(向殷皇子大馬路方向)',
    '殷皇子大馬路與約翰四世大馬路交界',
    '殷皇子大馬路與葡京路交界',
    '亞馬喇前地',
    '亞馬喇前地之圓形地方向',
    '亞馬喇前地巴士站',
    '水坑尾',
    '高偉樂街與荷蘭園大馬路交界',
    '美副將大馬路與士多鳥拜斯大馬路交界',
    '捐血中心',
    '華士古停車場',
    '宋玉生廣場',
    '羅理基博士大馬路行車隧道往松山隧道方向',
    '提督馬路與高士德大馬路交界（向高士德大馬路方向）',
    '提督馬路與高士德大馬路交界',
    '提督馬路與罅些喇海軍上將巷交界',
    '高士德與俾利喇街交界',
    '美副將大馬路與俾利喇街交界',
    '美副將大馬路與連勝馬路交界',
    '美副將大馬路與荷蘭園大馬路交界',
    '馬交石斜坡與俾利喇街交界',
    '沙梨頭海邊街',
    '沙梨頭海邊街與爹美刁施拿地大馬路交界',
    '沙梨頭海邊街與林茂巷交界',
    '沙梨頭海邊街與魚鰓巷交界(向新馬路方向)',
    '沙梨頭南街',
    '爹美刁斯拿地大馬路近栢港停車場',
    '爹美刁施拿地大馬路與魚鱗巷交界(向十六浦方向)',
    '巴素打爾古街近栢港停車場出口(向火船頭街方向)',
    '河邊新街與航海學校街交界(向媽閣方向)',
    '河邊新街與比厘喇馬忌士街交界(向媽閣廟方向)',
    '河邊新街與鹽巷交界（向媽閣方向）',
    '比厘喇馬忌士街與馬博士巷交界',
    '比厘喇馬忌士街與貨倉巷交界(向十六浦方向)',
    '火船頭街近11號碼頭（向巴素打爾古街方向)',
    '火船頭街（近11號碼頭）（向河邊新街方向）',
    '孫逸仙大馬路(近終審法院前地)',
    '西灣湖廣場(向孫逸仙大馬路)',
    '西灣湖廣場(向西灣湖景大馬路)',
    '孫逸仙大馬路與城市日大馬路交界',
    '西灣湖景大馬路(向孫逸仙大馬路方向)',
    '關閘廣場(向隧道方向)',
    '馬場北大馬路',
    '馬場北大馬路與馬場東大馬路交界',
    '巴坡沙大馬路與青洲大馬路交界',
    '青茂口岸-A',
    '青茂口岸-B',
    '黑沙環新街',
    '慕拉士大馬路',
    '勞動節大馬路',
    '東北大馬路與黑沙環中街交界向友誼圓形地方向',
    '友誼大馬路',
    '友誼大馬路(向馬揸度博士大馬路方向)',
    '馬揸度博士大馬路',
    '馬揸度博士大馬路(近勞工局)',
    '馬六甲街停車場',
    '友誼圓形地與友誼橋大馬路交界(向港珠澳大橋入口方向)',
    '友誼橋大馬路(向氹仔方向)',
    '外港客運碼頭 A',
    '外港客運碼頭 B',
    '松山隧道高士德方向',
    '松山隧道羅理基方向',
    '友誼大橋近港澳碼頭出口交匯處',
    '友誼大橋近澳門端之橋峰望澳門方向',
    '友誼大橋近氹仔端之橋峰望氹仔方向',
    '友誼大橋近北安入口交匯處',
    '友誼大橋近氹仔端出口',
    '西灣大橋澳門方向',
    '西灣大橋氹仔方向',
    '西灣大橋中段望澳門方向',
    '西灣大橋下層隧道(向澳門方向)',
    '東亞運圓形地(向西灣大橋入口)',
    '澳門大橋近人工島澳門口岸交匯處',
    '澳門大橋近澳門端入口',
    '澳門大橋近澳門端(向氹仔方向)',
    '澳門大橋中段澳門往氹仔方向',
    '澳門大橋中段氹仔往澳門方向',
    '澳門大橋近氹仔端出入口',
    '港珠澳邊檢北大馬路',
    '港珠澳邊檢南大馬路',
    '港珠澳邊檢南大馬路 (入境車道)',
    '馬萬祺博士大馬路（向友誼圓形地方向）',
    '泰安大馬路與馬萬祺博士大馬路交界',
    '鏡海大馬路',
    'A2橋澳門出口',
    '澳門橋大馬路與泰安大馬路交界',
    '港珠澳人工島往珠海通關車道A',
    '港珠澳人工島往珠海通關車道B',
    '港珠澳人工島往澳門通關車道A',
    '港珠澳人工島往澳門通關車道B',
    '北安圓形地',
    '東亞運街往東亞運大馬路方向',
    '奧林匹克游泳館圓形地往望德聖母灣大馬路方向',
    '路氹連貫公路圓形地',
    '北安大馬路往澳門方向',
    '偉龍馬路與飛機場圓形地交界',
    '路氹連貫公路下層隧道',
    '蘇利安圓形地',
    '海灣花園',
    '威尼斯人',
    '路氹連貫公路(近順榮大馬路)',
    '橫琴口岸澳門口岸區平台層 A',
    '橫琴口岸澳門口岸區平台層 B',
    '橫琴口岸澳門口岸區平台層 C',
    '橫琴口岸澳門口岸區平台層 D'
]

ORIGINAL_COORDS = {
    '新馬路與南灣大馬路交界': (22.19225, 113.54085),
    '水坑尾街與南灣大馬路交界路口': (22.19273, 113.5434),
    '新馬路與巴素打爾古街交界': (22.19642, 113.53648),
    '新馬路(近議事亭前地)向南灣大馬路方向': (22.19234, 113.54081),
    '南灣大馬路與區華利前地交界': (22.19105, 113.53927),
    '南灣大馬路與殷皇子大馬路(向八角亭方向)': (22.19224, 113.54096),
    '南灣大馬路與殷皇子大馬路(向殷皇子大馬路方向)': (22.19219, 113.54089),
    '殷皇子大馬路與約翰四世大馬路交界': (22.19111, 113.54202),
    '殷皇子大馬路與葡京路交界': (22.18993, 113.54341),
    '亞馬喇前地': (22.1899, 113.54339),
    '亞馬喇前地之圓形地方向': (22.18895, 113.54317),
    '亞馬喇前地巴士站': (22.18944, 113.54336),
    '水坑尾': (22.19405, 113.54369),
    '高偉樂街與荷蘭園大馬路交界': (22.19768, 113.54638),
    '美副將大馬路與士多鳥拜斯大馬路交界': (22.20191, 113.55247),
    '捐血中心': (22.18946, 113.54977),
    '華士古停車場': (22.19639, 113.5465),
    '宋玉生廣場': (22.19006, 113.55043),
    '羅理基博士大馬路行車隧道往松山隧道方向': (22.19619, 113.55196),
    '提督馬路與高士德大馬路交界（向高士德大馬路方向）': (22.20533, 113.54436),
    '提督馬路與高士德大馬路交界': (22.20525, 113.54427),
    '提督馬路與罅些喇海軍上將巷交界': (22.20401, 113.54255),
    '高士德與俾利喇街交界': (22.2029, 113.54709),
    '美副將大馬路與俾利喇街交界': (22.20496, 113.54877),
    '美副將大馬路與連勝馬路交界': (22.20625, 113.54715),
    '美副將大馬路與荷蘭園大馬路交界': (22.2028, 113.5498),
    '馬交石斜坡與俾利喇街交界': (22.20669, 113.55032),
    '沙梨頭海邊街': (22.20314, 113.54033),
    '沙梨頭海邊街與爹美刁施拿地大馬路交界': (22.20248, 113.53801),
    '沙梨頭海邊街與林茂巷交界': (22.2025, 113.53806),
    '沙梨頭海邊街與魚鰓巷交界(向新馬路方向)': (22.20018, 113.53728),
    '沙梨頭南街': (22.206, 113.5405),
    '爹美刁斯拿地大馬路近栢港停車場': (22.19829, 113.53657),
    '爹美刁施拿地大馬路與魚鱗巷交界(向十六浦方向)': (22.20174, 113.53727),
    '巴素打爾古街近栢港停車場出口(向火船頭街方向)': (22.19765, 113.53669),
    '河邊新街與航海學校街交界(向媽閣方向)': (22.18746, 113.5312),
    '河邊新街與比厘喇馬忌士街交界(向媽閣廟方向)': (22.18947, 113.53268),
    '河邊新街與鹽巷交界（向媽閣方向）': (22.19011, 113.53309),
    '比厘喇馬忌士街與馬博士巷交界': (22.19218, 113.534),
    '比厘喇馬忌士街與貨倉巷交界(向十六浦方向)': (22.19108, 113.53459),
    '火船頭街近11號碼頭（向巴素打爾古街方向)': (22.1963, 113.53647),
    '火船頭街（近11號碼頭）（向河邊新街方向）': (22.19319, 113.5352),
    '孫逸仙大馬路(近終審法院前地)': (22.18239, 113.54111),
    '西灣湖廣場(向孫逸仙大馬路)': (22.18146, 113.53839),
    '西灣湖廣場(向西灣湖景大馬路)': (22.18044, 113.53762),
    '孫逸仙大馬路與城市日大馬路交界': (22.18585, 113.54943),
    '西灣湖景大馬路(向孫逸仙大馬路方向)': (22.1808, 113.537),
    '關閘廣場(向隧道方向)': (22.2152, 113.5501),
    '馬場北大馬路': (22.2135, 113.5525),
    '馬場北大馬路與馬場東大馬路交界': (22.21385, 113.55444),
    '巴坡沙大馬路與青洲大馬路交界': (22.21027, 113.54734),
    '青茂口岸-A': (22.2145, 113.546),
    '青茂口岸-B': (22.2145, 113.546),
    '黑沙環新街': (22.20613, 113.55738),
    '慕拉士大馬路': (22.20665, 113.55256),
    '勞動節大馬路': (22.21059, 113.55409),
    '東北大馬路與黑沙環中街交界向友誼圓形地方向': (22.20938, 113.55765),
    '友誼大馬路': (22.19077, 113.5487),
    '友誼大馬路(向馬揸度博士大馬路方向)': (22.20474, 113.5606),
    '馬揸度博士大馬路': (22.20563, 113.55812),
    '馬揸度博士大馬路(近勞工局)': (22.20573, 113.55884),
    '馬六甲街停車場': (22.19606, 113.55367),
    '友誼圓形地與友誼橋大馬路交界(向港珠澳大橋入口方向)': (22.21175, 113.55951),
    '友誼橋大馬路(向氹仔方向)': (22.20866, 113.56081),
    '外港客運碼頭 A': (22.197, 113.5575),
    '外港客運碼頭 B': (22.197, 113.5575),
    '松山隧道高士德方向': (22.19889, 113.55073),
    '松山隧道羅理基方向': (22.19674, 113.55192),
    '友誼大橋近港澳碼頭出口交匯處': (22.198, 113.5605),
    '友誼大橋近澳門端之橋峰望澳門方向': (22.199, 113.5625),
    '友誼大橋近氹仔端之橋峰望氹仔方向': (22.167, 113.567),
    '友誼大橋近北安入口交匯處': (22.159, 113.5685),
    '友誼大橋近氹仔端出口': (22.1575, 113.569),
    '西灣大橋澳門方向': (22.178, 113.5365),
    '西灣大橋氹仔方向': (22.15, 113.54),
    '西灣大橋中段望澳門方向': (22.17, 113.5365),
    '西灣大橋下層隧道(向澳門方向)': (22.17, 113.536),
    '東亞運圓形地(向西灣大橋入口)': (22.151, 113.543),
    '澳門大橋近人工島澳門口岸交匯處': (22.2055, 113.5745),
    '澳門大橋近澳門端入口': (22.207, 113.567),
    '澳門大橋近澳門端(向氹仔方向)': (22.206, 113.568),
    '澳門大橋中段澳門往氹仔方向': (22.188, 113.572),
    '澳門大橋中段氹仔往澳門方向': (22.185, 113.573),
    '澳門大橋近氹仔端出入口': (22.165, 113.5725),
    '港珠澳邊檢北大馬路': (22.208, 113.575),
    '港珠澳邊檢南大馬路': (22.2035, 113.5755),
    '港珠澳邊檢南大馬路 (入境車道)': (22.2035, 113.576),
    '馬萬祺博士大馬路（向友誼圓形地方向）': (22.2105, 113.562),
    '泰安大馬路與馬萬祺博士大馬路交界': (22.2105, 113.565),
    '鏡海大馬路': (22.2085, 113.566),
    'A2橋澳門出口': (22.212, 113.555),
    '澳門橋大馬路與泰安大馬路交界': (22.212, 113.567),
    '港珠澳人工島往珠海通關車道A': (22.205, 113.5775),
    '港珠澳人工島往珠海通關車道B': (22.204, 113.5775),
    '港珠澳人工島往澳門通關車道A': (22.205, 113.576),
    '港珠澳人工島往澳門通關車道B': (22.204, 113.576),
    '北安圓形地': (22.158, 113.568),
    '東亞運街往東亞運大馬路方向': (22.152, 113.55),
    '奧林匹克游泳館圓形地往望德聖母灣大馬路方向': (22.152, 113.558),
    '路氹連貫公路圓形地': (22.15, 113.562),
    '北安大馬路往澳門方向': (22.1575, 113.57),
    '偉龍馬路與飛機場圓形地交界': (22.153, 113.572),
    '路氹連貫公路下層隧道': (22.14, 113.562),
    '蘇利安圓形地': (22.155, 113.548),
    '海灣花園': (22.158, 113.553),
    '威尼斯人': (22.148, 113.562),
    '路氹連貫公路(近順榮大馬路)': (22.138, 113.563),
    '橫琴口岸澳門口岸區平台層 A': (22.1395, 113.545),
    '橫琴口岸澳門口岸區平台層 B': (22.1395, 113.5455),
    '橫琴口岸澳門口岸區平台層 C': (22.14, 113.545),
    '橫琴口岸澳門口岸區平台層 D': (22.14, 113.5455),
}

CAM_TAGS = {
    '新馬路與南灣大馬路交界': ['peninsula'],
    '水坑尾街與南灣大馬路交界路口': ['peninsula'],
    '新馬路與巴素打爾古街交界': ['peninsula'],
    '新馬路(近議事亭前地)向南灣大馬路方向': ['peninsula'],
    '南灣大馬路與區華利前地交界': ['peninsula'],
    '南灣大馬路與殷皇子大馬路(向八角亭方向)': ['peninsula'],
    '南灣大馬路與殷皇子大馬路(向殷皇子大馬路方向)': ['peninsula'],
    '殷皇子大馬路與約翰四世大馬路交界': ['peninsula'],
    '殷皇子大馬路與葡京路交界': ['peninsula'],
    '亞馬喇前地': ['peninsula'],
    '亞馬喇前地之圓形地方向': ['peninsula'],
    '亞馬喇前地巴士站': ['peninsula'],
    '水坑尾': ['peninsula'],
    '高偉樂街與荷蘭園大馬路交界': ['peninsula'],
    '美副將大馬路與士多鳥拜斯大馬路交界': ['peninsula'],
    '捐血中心': ['peninsula'],
    '華士古停車場': ['peninsula'],
    '宋玉生廣場': ['peninsula'],
    '羅理基博士大馬路行車隧道往松山隧道方向': ['peninsula'],
    '提督馬路與高士德大馬路交界（向高士德大馬路方向）': ['peninsula'],
    '提督馬路與高士德大馬路交界': ['peninsula'],
    '提督馬路與罅些喇海軍上將巷交界': ['peninsula'],
    '高士德與俾利喇街交界': ['peninsula'],
    '美副將大馬路與俾利喇街交界': ['peninsula'],
    '美副將大馬路與連勝馬路交界': ['peninsula'],
    '美副將大馬路與荷蘭園大馬路交界': ['peninsula'],
    '馬交石斜坡與俾利喇街交界': ['peninsula'],
    '沙梨頭海邊街': ['peninsula'],
    '沙梨頭海邊街與爹美刁施拿地大馬路交界': ['peninsula'],
    '沙梨頭海邊街與林茂巷交界': ['peninsula'],
    '沙梨頭海邊街與魚鰓巷交界(向新馬路方向)': ['peninsula'],
    '沙梨頭南街': ['peninsula'],
    '爹美刁斯拿地大馬路近栢港停車場': ['peninsula'],
    '爹美刁施拿地大馬路與魚鱗巷交界(向十六浦方向)': ['peninsula'],
    '巴素打爾古街近栢港停車場出口(向火船頭街方向)': ['peninsula'],
    '河邊新街與航海學校街交界(向媽閣方向)': ['peninsula'],
    '河邊新街與比厘喇馬忌士街交界(向媽閣廟方向)': ['peninsula'],
    '河邊新街與鹽巷交界（向媽閣方向）': ['peninsula'],
    '比厘喇馬忌士街與馬博士巷交界': ['peninsula'],
    '比厘喇馬忌士街與貨倉巷交界(向十六浦方向)': ['peninsula'],
    '火船頭街近11號碼頭（向巴素打爾古街方向)': ['peninsula'],
    '火船頭街（近11號碼頭）（向河邊新街方向）': ['peninsula'],
    '孫逸仙大馬路(近終審法院前地)': ['peninsula'],
    '西灣湖廣場(向孫逸仙大馬路)': ['peninsula'],
    '西灣湖廣場(向西灣湖景大馬路)': ['peninsula'],
    '孫逸仙大馬路與城市日大馬路交界': ['peninsula'],
    '西灣湖景大馬路(向孫逸仙大馬路方向)': ['bridge'],
    '關閘廣場(向隧道方向)': ['peninsula'],
    '馬場北大馬路': ['peninsula'],
    '馬場北大馬路與馬場東大馬路交界': ['peninsula'],
    '巴坡沙大馬路與青洲大馬路交界': ['peninsula'],
    '青茂口岸-A': ['peninsula'],
    '青茂口岸-B': ['peninsula'],
    '黑沙環新街': ['peninsula'],
    '慕拉士大馬路': ['peninsula'],
    '勞動節大馬路': ['peninsula'],
    '東北大馬路與黑沙環中街交界向友誼圓形地方向': ['peninsula'],
    '友誼大馬路': ['peninsula'],
    '友誼大馬路(向馬揸度博士大馬路方向)': ['peninsula'],
    '馬揸度博士大馬路': ['peninsula'],
    '馬揸度博士大馬路(近勞工局)': ['peninsula'],
    '馬六甲街停車場': ['peninsula'],
    '友誼圓形地與友誼橋大馬路交界(向港珠澳大橋入口方向)': ['peninsula'],
    '友誼橋大馬路(向氹仔方向)': ['peninsula'],
    '外港客運碼頭 A': ['peninsula'],
    '外港客運碼頭 B': ['peninsula'],
    '松山隧道高士德方向': ['tunnel'],
    '松山隧道羅理基方向': ['tunnel'],
    '友誼大橋近港澳碼頭出口交匯處': ['bridge'],
    '友誼大橋近澳門端之橋峰望澳門方向': ['bridge'],
    '友誼大橋近氹仔端之橋峰望氹仔方向': ['bridge'],
    '友誼大橋近北安入口交匯處': ['bridge'],
    '友誼大橋近氹仔端出口': ['bridge'],
    '西灣大橋澳門方向': ['bridge'],
    '西灣大橋氹仔方向': ['bridge'],
    '西灣大橋中段望澳門方向': ['bridge'],
    '西灣大橋下層隧道(向澳門方向)': ['bridge'],
    '東亞運圓形地(向西灣大橋入口)': ['bridge'],
    '澳門大橋近人工島澳門口岸交匯處': ['bridge'],
    '澳門大橋近澳門端入口': ['bridge'],
    '澳門大橋近澳門端(向氹仔方向)': ['bridge'],
    '澳門大橋中段澳門往氹仔方向': ['bridge'],
    '澳門大橋中段氹仔往澳門方向': ['bridge'],
    '澳門大橋近氹仔端出入口': ['bridge'],
    '港珠澳邊檢北大馬路': ['border'],
    '港珠澳邊檢南大馬路': ['border'],
    '港珠澳邊檢南大馬路 (入境車道)': ['border'],
    '馬萬祺博士大馬路（向友誼圓形地方向）': ['link_road'],
    '泰安大馬路與馬萬祺博士大馬路交界': ['link_road'],
    '鏡海大馬路': ['link_road'],
    'A2橋澳門出口': ['link_road'],
    '澳門橋大馬路與泰安大馬路交界': ['link_road'],
    '港珠澳人工島往珠海通關車道A': ['border', 'artificial_island'],
    '港珠澳人工島往珠海通關車道B': ['border', 'artificial_island'],
    '港珠澳人工島往澳門通關車道A': ['border', 'artificial_island'],
    '港珠澳人工島往澳門通關車道B': ['border', 'artificial_island'],
    '北安圓形地': ['taipa'],
    '東亞運街往東亞運大馬路方向': ['taipa'],
    '奧林匹克游泳館圓形地往望德聖母灣大馬路方向': ['taipa'],
    '路氹連貫公路圓形地': ['taipa'],
    '北安大馬路往澳門方向': ['taipa'],
    '偉龍馬路與飛機場圓形地交界': ['taipa'],
    '路氹連貫公路下層隧道': ['tunnel'],
    '蘇利安圓形地': ['taipa'],
    '海灣花園': ['taipa'],
    '威尼斯人': ['taipa'],
    '路氹連貫公路(近順榮大馬路)': ['taipa'],
    '橫琴口岸澳門口岸區平台層 A': ['port'],
    '橫琴口岸澳門口岸區平台層 B': ['port'],
    '橫琴口岸澳門口岸區平台層 C': ['port'],
    '橫琴口岸澳門口岸區平台層 D': ['port'],
}

def resolve_overlaps(coords_dict, min_dist=0.00015):
    coords = {k: list(v) if v else [0, 0] for k, v in coords_dict.items()}
    keys = list(coords.keys())
    for _ in range(15):
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                lat1, lon1 = coords[k1]; lat2, lon2 = coords[k2]
                dist = math.hypot(lat1 - lat2, lon1 - lon2)
                if dist < min_dist:
                    angle = math.atan2(lat1 - lat2, lon1 - lon2) if dist != 0 else i * 0.5
                    push = (min_dist - (dist if dist != 0 else 0.00001)) / 2.0
                    coords[k1][0] += push * math.sin(angle); coords[k1][1] += push * math.cos(angle)
                    coords[k2][0] -= push * math.sin(angle); coords[k2][1] -= push * math.cos(angle)
    return {k: tuple(v) for k, v in coords.items()}

_valid_coords = {k: v for k, v in ORIGINAL_COORDS.items() if v is not None}
_resolved = resolve_overlaps(_valid_coords)
STREET_COORDS = {k: (_resolved.get(k) or ORIGINAL_COORDS.get(k)) for k in ORIGINAL_COORDS}

# =========================
# 攝影機類型統計
# =========================
def print_camera_stats():
    """印出所有攝影機的類型分類統計"""
    from collections import Counter
    tag_counts = Counter()
    for loc, tags in CAM_TAGS.items():
        for t in tags:
            tag_counts[t] += 1
    tag_names = {
        'peninsula': '澳門半島道路', 'taipa': '氹仔/路環',
        'bridge': '跨海大橋', 'tunnel': '隧道',
        'port': '口岸', 'border': '邊檢/通關',
        'link_road': '港珠澳連接路', 'artificial_island': '人工島'
    }
    print("=" * 50)
    print(f"v2 攝影機統計: 共 {len(MACAU_STREETS)} 支")
    for tag, count in tag_counts.most_common():
        print(f"  {tag_names.get(tag, tag)}: {count} 支")
    with_coords = sum(1 for v in ORIGINAL_COORDS.values() if v is not None)
    print(f"  有GPS座標: {with_coords} 支")
    print(f"  無GPS座標(估算): {len(ORIGINAL_COORDS) - with_coords} 支")
    print(f"  特殊覆蓋設定: {len(CAM_OVERRIDES)} 支")
    print("=" * 50)

# =========================
# 雲端資料推播 (全澳門路網)
# =========================
MACAU_GRAPH = None

# 全澳門範圍 (澳門半島 + 氹仔 + 路環 + 港珠澳人工島)
MACAU_BBOX = (113.52, 22.10, 113.60, 22.22)

def preload_macau_graph():
    global MACAU_GRAPH
    if ox is None:
        return
    try:
        print(f"下載全澳門路網 (含氹仔/路環)...")
        MACAU_GRAPH = ox.graph_from_bbox(
            bbox=MACAU_BBOX, network_type='drive')
        print(f"全澳門路網載入成功！節點: {len(MACAU_GRAPH.nodes)}, 邊: {len(MACAU_GRAPH.edges)}")
    except Exception as e:
        print(f"路網載入失敗: {e}")

def run_heat_diffusion(G, source_scores, iterations=40):
    scores = {n: None for n in G.nodes()}
    for n, s in source_scores.items():
        scores[n] = s
    for _ in range(iterations):
        new_scores = scores.copy()
        for n in G.nodes():
            if n in source_scores:
                continue
            neighbors = list(G.successors(n)) + list(G.predecessors(n))
            vals = [scores[nb] for nb in neighbors if scores[nb] is not None]
            if vals:
                new_scores[n] = sum(vals) / len(vals)
        scores = new_scores
    for n in G.nodes():
        if scores[n] is None:
            scores[n] = 0.0
    return scores

def post_macau_traffic_data(states, cloud_url):
    if MACAU_GRAPH is None:
        return
    try:
        current_scores = {
            st.name: (st.last_metrics.get("jam_avg", 0.0) if st.last_metrics else 0.0)
            for st in states
        }

        def get_color(score):
            if score < 0.3: return "#28a745"
            elif score < 0.75: return "#ffc107"
            else: return "#dc3545"

        # 將有座標的攝影機映射到最近的路網節點
        camera_node_scores = {}
        mapped_count = 0
        for name, coord in STREET_COORDS.items():
            if name in current_scores and coord is not None:
                try:
                    nearest_node = ox.distance.nearest_nodes(
                        MACAU_GRAPH, X=coord[1], Y=coord[0])
                    camera_node_scores[nearest_node] = current_scores[name]
                    mapped_count += 1
                except Exception:
                    pass  # 節點可能超出圖範圍

        # 熱擴散: 將攝影機數據沿路網傳播
        node_scores = run_heat_diffusion(MACAU_GRAPH, camera_node_scores)

        # 建立道路 GeoJSON
        features = []
        for u, v, k, data in MACAU_GRAPH.edges(keys=True, data=True):
            sc = (node_scores[u] + node_scores[v]) / 2.0
            if 'geometry' in data:
                coords_data = [[lon, lat] for lon, lat in data['geometry'].coords]
            else:
                coords_data = [
                    [MACAU_GRAPH.nodes[u]['x'], MACAU_GRAPH.nodes[u]['y']],
                    [MACAU_GRAPH.nodes[v]['x'], MACAU_GRAPH.nodes[v]['y']],
                ]
            # 只輸出有交通數據的道路 (減少資料量)
            if sc > 0.01:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords_data},
                    "properties": {
                        "type": "street",
                        "score": round(sc, 3),
                        "color": get_color(sc),
                    },
                })

        # 建立攝影機節點 GeoJSON (全部 111 個, 包含無座標的)
        for st in states:
            coord = STREET_COORDS.get(st.name)
            score = current_scores.get(st.name, 0.0)
            tags = CAM_TAGS.get(st.name, [])
            props = {
                "type": "camera",
                "name": st.name,
                "score": round(score, 3),
                "color": get_color(score),
                "veh_active": st.last_metrics.get("veh_active", 0) if st.last_metrics else 0,
                "flow_total": st.last_metrics.get("flow_total_vpm", 0) if st.last_metrics else 0,
                "stream_url": st.stream_url,
                "tags": tags,
            }
            if coord is not None:
                lat, lon = coord
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": props,
                })

        payload = {
            "type": "FeatureCollection",
            "features": features,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        requests.post(cloud_url, json=payload, timeout=2.0)
    except Exception as e:
        print(f"雲端推播錯誤: {e}")

# =========================
# 追蹤與影像處理
# =========================
@dataclass
class Track:
    tid: int; bbox: np.ndarray; conf: float; cls_id: int
    hits: int = 1; time_since_update: int = 0; last_t: float = 0.0
    center: Optional[Tuple[float, float]] = None; speed_ema_px_s: float = 0.0
    stationary_time: float = 0.0
    prev_center: Optional[Tuple[float, float]] = None
    counted: bool = False

class SimpleIOUTracker:
    def __init__(self, iou_th=0.3, max_age=6):
        self.iou_th, self.max_age, self._next_id = iou_th, max_age, 1
        self.tracks: list[Track] = []

    def update(self, detections, t, fw, fh, stationary_th=2.0):
        for tr in self.tracks:
            tr.time_since_update += 1
        matches = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                iou = iou_xyxy(tr.bbox, det["bbox"])
                if iou >= self.iou_th:
                    matches.append((iou, ti, di))
        matches.sort(key=lambda x: x[0], reverse=True)
        u_tr, u_det = set(), set()
        for iou, ti, di in matches:
            if ti in u_tr or di in u_det:
                continue
            u_tr.add(ti); u_det.add(di)
            tr, det = self.tracks[ti], detections[di]
            prev_c = tr.center if tr.center else (
                float((tr.bbox[0] + tr.bbox[2]) / 2),
                float((tr.bbox[1] + tr.bbox[3]) / 2),
            )
            tr.bbox = det["bbox"]
            tr.center = (
                float((det["bbox"][0] + det["bbox"][2]) / 2),
                float((det["bbox"][1] + det["bbox"][3]) / 2),
            )
            tr.prev_center = prev_c
            dt = max(1e-3, t - tr.last_t)
            speed = math.hypot(tr.center[0] - prev_c[0], tr.center[1] - prev_c[1]) / dt
            tr.speed_ema_px_s = 0.6 * tr.speed_ema_px_s + 0.4 * speed
            if tr.speed_ema_px_s < stationary_th:
                tr.stationary_time += dt
            else:
                tr.stationary_time = 0.0
            tr.last_t, tr.hits, tr.time_since_update = float(t), tr.hits + 1, 0
        for di, det in enumerate(detections):
            if di not in u_det:
                self.tracks.append(Track(
                    self._next_id, det["bbox"], det["conf"], det["cls_id"],
                    last_t=t,
                    center=(float((det["bbox"][0] + det["bbox"][2]) / 2),
                            float((det["bbox"][1] + det["bbox"][3]) / 2)),
                ))
                self._next_id += 1
        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]
        return [tr for tr in self.tracks if tr.time_since_update == 0]

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
        if self.name in ["亞馬喇前地之圓形地方向", "亞馬喇前地巴士站"]:
            self.use_motion = True
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=30, detectShadows=False)

def draw_chinese_text_bg(img, text, x, y, font_size=16, fg=(255, 255, 255), bg=(0, 0, 0)):
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=bg)
    draw.text((x, y), text, font=font, fill=fg)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# =========================
# 批次推論與後處理
# =========================
def _preprocess_frame(frame, ovr):
    if "crop" in ovr:
        x1, y1, x2, y2 = ovr["crop"]
        h_orig, w_orig = frame.shape[:2]
        y_s, y_e, x_s, x_e = max(0, y1), min(h_orig, y2), max(0, x1), min(w_orig, x2)
        if y_e > y_s and x_e > x_s:
            frame = frame[y_s:y_e, x_s:x_e]
    proc_size = ovr.get("proc_size", (PROC_W, PROC_H))
    if proc_size is None:
        proc = frame.copy()
    else:
        proc = cv2.resize(frame, proc_size)
    if proc is None or proc.size == 0:
        return None
    fh, fw = proc.shape[:2]
    speed_scale = ovr.get("speed_scale", 1.0)
    capacity_scale = ovr.get("capacity_scale", 1.0)
    imgsz = ovr.get("imgsz", IMGSZ)
    return proc, fw, fh, speed_scale, capacity_scale, imgsz

def _infer_batch(model, batch_items):
    if not batch_items:
        return []
    groups = {}
    for i, (proc, imgsz) in enumerate(batch_items):
        groups.setdefault(imgsz, []).append(i)
    results = [None] * len(batch_items)
    for imgsz, indices in groups.items():
        frames = [batch_items[j][0] for j in indices]
        batch_results = model.predict(
            frames, imgsz=imgsz, conf=CONF, classes=INCLUDE_CLASSES,
            device=DEVICE, half=HALF, verbose=False,
        )
        for j, res in zip(indices, batch_results):
            results[j] = res
    return results

def _extract_detections(result):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    dets = []
    for i in range(len(result.boxes)):
        dets.append({
            "bbox": result.boxes.xyxy[i].cpu().numpy(),
            "conf": float(result.boxes.conf[i]),
            "cls_id": int(result.boxes.cls[i]),
        })
    return dets

def _motion_supplement(st, proc, dets):
    if not st.use_motion or st.bg_subtractor is None:
        return dets
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    fg_mask = st.bg_subtractor.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 8000:
            x, y, w, h = cv2.boundingRect(cnt)
            m_bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            if not any(iou_xyxy(m_bbox, d["bbox"]) > 0.1 for d in dets):
                dets.append({"bbox": m_bbox, "conf": 0.5, "cls_id": 5})
    return dets

def _track_and_score(st, dets, now, fw, fh, stat_spd_th, stop_spd_th,
                     norm_spd_th, capacity_scale, jam_occ_th, max_car_area):
    tracks = st.tracker.update(dets, now, fw, fh, stationary_th=stat_spd_th)
    stable = [t for t in tracks if t.hits >= TRACK_MIN_HITS]
    active_vehicles = [t for t in stable if t.stationary_time < PARKED_TIME_SEC]
    parked_vehicles = [t for t in stable if t.stationary_time >= PARKED_TIME_SEC]

    v_act = len(active_vehicles)
    st.veh_history.append(v_act)
    history_list = list(st.veh_history)[-50:]
    median_v = sorted(history_list)[len(history_list) // 2] if len(history_list) > 10 else 0

    occ = sum(min((t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]), max_car_area)
              for t in active_vehicles) / (fw * fh) if fw * fh > 0 else 0
    spd = np.median([t.speed_ema_px_s for t in active_vehicles]) if active_vehicles else 0
    stop = sum(1 for t in active_vehicles if t.speed_ema_px_s < stop_spd_th) / v_act if v_act > 0 else 0

    line_y = int(fh * LINE_Y_RATIO)
    for tr in active_vehicles:
        if tr.prev_center and tr.center and not tr.counted:
            py, cy = tr.prev_center[1], tr.center[1]
            if (py < line_y and cy >= line_y) or (py > line_y and cy <= line_y):
                st.flow_timestamps.append(now)
                tr.counted = True
    while st.flow_timestamps and now - st.flow_timestamps[0] > 60.0:
        st.flow_timestamps.popleft()
    current_vpm = len(st.flow_timestamps)

    if v_act == 0:
        raw_jam = 0.0
        for _ in range(5):
            st.jam_history.append(0.0)
    else:
        raw_jam = (0.30 * min(1, occ / jam_occ_th) +
                   0.20 * (1 - min(1, spd / norm_spd_th)) +
                   0.50 * stop)
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
    return active_vehicles, parked_vehicles, v_act, jam_avg, current_vpm, line_y

def _render_tile(st, proc, active_vehicles, parked_vehicles, v_act,
                 jam_avg, current_vpm, line_y):
    anno = proc.copy()
    fh, fw = proc.shape[:2]
    cv2.line(anno, (0, line_y), (fw, line_y), (255, 0, 255), 2)
    for tr in active_vehicles:
        cv2.rectangle(anno, (int(tr.bbox[0]), int(tr.bbox[1])),
                      (int(tr.bbox[2]), int(tr.bbox[3])), (0, 255, 0), 2)
    for tr in parked_vehicles:
        cv2.rectangle(anno, (int(tr.bbox[0]), int(tr.bbox[1])),
                      (int(tr.bbox[2]), int(tr.bbox[3])), (0, 165, 255), 2)
    anno = cv2.resize(anno, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)
    bar_color = ((40, 167, 69) if jam_avg < 0.4 else
                 (7, 193, 255) if jam_avg < 0.7 else (53, 53, 220))
    cv2.rectangle(anno, (0, TILE_H - 8), (int(TILE_W * jam_avg), TILE_H), bar_color, -1)

    # v2: 類型標記
    tags = CAM_TAGS.get(st.name, [])
    tag_icon = ""
    if "bridge" in tags: tag_icon = "B:"
    elif "tunnel" in tags: tag_icon = "T:"
    elif "port" in tags: tag_icon = "P:"
    elif "border" in tags: tag_icon = "C:"
    elif "taipa" in tags: tag_icon = "I:"

    anno = draw_chinese_text_bg(
        anno, f"{tag_icon}{st.name} 車:{v_act} 流/分:{current_vpm}",
        5, 5, font_size=12, bg=bar_color)
    st.last_tile = anno

# =========================
# 主迴圈
# =========================
def main():
    if not os.path.exists(STREAMS_FILE):
        print(f"找不到串流檔案: {STREAMS_FILE}")
        return

    streams = []
    seen = set()
    with open(STREAMS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # 支援 "URL | cam_id | 地點 | 座標" 格式，只取 URL 部分
            if "|" in s:
                s = s.split("|")[0].strip()
            if s.startswith("https://") and s not in seen:
                streams.append(s)
                seen.add(s)

    streams = streams[:MAX_CAMERAS]

    print(f"v2 已載入 {len(streams)} 支 CCTV 串流")
    print_camera_stats()

    # --- Step 1: 先載入 YOLO 模型 (避免鏡頭卡住時連模型都沒載) ---
    print(f"載入 YOLO 模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"YOLO 模型已載入 (device={DEVICE}, half={HALF}, imgsz={IMGSZ})")

    # --- Step 2: 載入路網 (背景執行) ---
    threading.Thread(target=preload_macau_graph, daemon=True).start()

    # --- Step 3: 背景逐步載入全部鏡頭 ---
    print(f"HTTP 預檢 {len(streams)} 個串流...")
    working_urls = []
    t_check = time.time()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for i, url in enumerate(streams):
        try:
            r = session.get(url, timeout=1.5, stream=True)
            if r.status_code == 200:
                working_urls.append((i, url))
                r.close()
        except Exception:
            pass

    print(f"  HTTP OK: {len(working_urls)}/{len(streams)} ({time.time() - t_check:.1f}s)")

    if len(working_urls) == 0:
        print("所有 URL 無回應, 請檢查網路連線")
        test_url = streams[0] if streams else ""
        print(f"  測試: {test_url}")
        try:
            r = requests.get(test_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            print(f"  Status: {r.status_code}")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")
        return

    # --- Step 3: 背景逐步載入全部鏡頭 ---
    cameras = []       # StreamCamera 物件
    states = []        # CameraState 物件
    camera_lock = threading.Lock()

    def _open_one(name, url):
        """嘗試開啟一個鏡頭, 回傳 (cam, state) 或 (None, None)"""
        result = [None]
        ready = threading.Event()
        def _target():
            try:
                result[0] = StreamCamera(url, name)
            except Exception:
                pass
            finally:
                ready.set()
        t = threading.Thread(target=_target, daemon=True)
        t.start()
        ready.wait(timeout=3.0)
        if result[0] is not None:
            return result[0], CameraState(name=name, tracker=SimpleIOUTracker(), stream_url=url)
        return None, None

    # 先同步開第一批 (10 個) 讓 UI 盡快有畫面
    first_batch = working_urls[:10]
    rest = working_urls[10:]
    print(f"啟動首批 {len(first_batch)} 個鏡頭...")
    t0 = time.time()
    for idx, url in first_batch:
        name = MACAU_STREETS[idx] if idx < len(MACAU_STREETS) else f"Cam {idx + 1}"
        cam, st = _open_one(name, url)
        if cam is not None:
            cameras.append(cam)
            states.append(st)
    print(f"  首批 OK: {len(cameras)} 個 ({time.time() - t0:.0f}s)")

    # 背景載入其餘鏡頭
    if rest:
        print(f"  背景載入其餘 {len(rest)} 個...")
        def _load_rest():
            loaded = 0
            for idx, url in rest:
                name = MACAU_STREETS[idx] if idx < len(MACAU_STREETS) else f"Cam {idx + 1}"
                cam, st = _open_one(name, url)
                if cam is not None:
                    with camera_lock:
                        cameras.append(cam)
                        states.append(st)
                    loaded += 1
            print(f"  [載入完成] 共 {loaded} 個鏡頭")
        threading.Thread(target=_load_rest, daemon=True).start()

    if len(cameras) == 0:
        print("FFmpeg 無法開啟任何串流")
        return

    last_map_update_t = time.time()
    last_ui_t = 0.0
    last_perf_t = time.time()
    total_frames_processed = 0

    if SHOW_UI:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        loop_start = time.time()

        # Phase 0: 定期雲端 + 本地推播
        if (loop_start - last_map_update_t) > 5.0:
            threading.Thread(
                target=post_macau_traffic_data,
                args=(states, CLOUD_API_URL), daemon=True,
            ).start()
            threading.Thread(
                target=post_macau_traffic_data,
                args=(states, LOCAL_API_URL), daemon=True,
            ).start()
            last_map_update_t = loop_start

        # Phase 1: 收集畫面
        batch_items = []
        batch_meta = []
        for idx in range(len(cameras)):
            st = states[idx]
            if st.last_proc_t > 0 and (loop_start - st.last_proc_t) < PROC_INTERVAL_SEC:
                continue
            ret, frame = cameras[idx].get_frame()
            if not ret or frame is None:
                continue
            st.last_proc_t = loop_start
            ovr = CAM_OVERRIDES.get(st.name, {})
            prepped = _preprocess_frame(frame, ovr)
            if prepped is None:
                continue
            proc, fw, fh, speed_scale, capacity_scale, imgsz = prepped
            batch_items.append((proc, imgsz))
            batch_meta.append((idx, proc, fw, fh, speed_scale, capacity_scale))

        # Phase 2: 批次 GPU 推論
        batch_results = _infer_batch(model, batch_items)
        total_frames_processed += len(batch_items)

        # Phase 3: 後處理
        for (st_idx, proc, fw, fh, speed_scale, capacity_scale), result in zip(
                batch_meta, batch_results):
            st = states[st_idx]
            dets = _extract_detections(result)
            dets = _motion_supplement(st, proc, dets)
            stat_spd_th = STATIONARY_SPEED_TH * speed_scale
            stop_spd_th = 3.0 * speed_scale
            norm_spd_th = 20.0 * speed_scale
            jam_occ_th = 0.40
            max_car_area = (fw * fh) * 0.10
            active_vehicles, parked_vehicles, v_act, jam_avg, current_vpm, line_y = \
                _track_and_score(
                    st, dets, loop_start, fw, fh, stat_spd_th, stop_spd_th,
                    norm_spd_th, capacity_scale, jam_occ_th, max_car_area)
            _render_tile(st, proc, active_vehicles, parked_vehicles, v_act,
                         jam_avg, current_vpm, line_y)

        # Phase 4: UI 網格
        if SHOW_UI and batch_items and (loop_start - last_ui_t) >= (1.0 / UI_FPS):
            last_ui_t = loop_start
            grid = np.zeros((GRID_ROWS * TILE_H, GRID_COLS * TILE_W, 3), dtype=np.uint8)
            for i, st in enumerate(states):
                if st.last_tile is not None and i < GRID_ROWS * GRID_COLS:
                    r, c = i // GRID_COLS, i % GRID_COLS
                    grid[r * TILE_H:(r + 1) * TILE_H,
                         c * TILE_W:(c + 1) * TILE_W] = st.last_tile
            cv2.imshow(WINDOW_NAME, grid)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 效能報告
        if (loop_start - last_perf_t) >= PERF_LOG_INTERVAL:
            elapsed = loop_start - last_perf_t
            fps_all = total_frames_processed / elapsed
            fps_per_cam = fps_all / max(1, len(cameras))
            active_cams = sum(1 for st in states if st.last_tile is not None)
            print(f"FPS: {fps_all:.1f} total ({fps_per_cam:.2f}/cam) | "
                  f"batch={len(batch_items):d} active={active_cams}/{len(cameras)} | "
                  f"GPU={torch.cuda.memory_reserved(0)/1e9:.1f}GB")
            total_frames_processed = 0
            last_perf_t = loop_start

        if not batch_items:
            time.sleep(0.005)

    for c in cameras:
        c.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
