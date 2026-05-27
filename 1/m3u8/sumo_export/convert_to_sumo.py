#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ============================================================================
# convert_to_sumo.py — 將 v2_detector 收集的訓練數據轉換為 SUMO .rou.xml
# ============================================================================
# 輸入: training_data/YYYY-MM-DD/edge_flow_*.jsonl + camera_metrics_*.csv
# 輸出: sumo_export/macao_traffic.rou.xml (SUMO 車流定義檔)
#
# 用法:
#   python convert_to_sumo.py                          # 轉換所有日期
#   python convert_to_sumo.py --date 2026-05-28        # 指定日期
#   python convert_to_sumo.py --hours 7-9,17-19        # 只轉尖峰時段
# ============================================================================

import os
import sys
import json
import csv
import re
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime

# --- 設定 ---
TRAINING_DATA_DIR = r"D:\school\bsd\1\m3u8\training_data"
SUMO_NET_FILE = r"D:\school\bsd\SUMO\macao.net.xml"
SUMO_OUTPUT_DIR = r"D:\school\bsd\1\m3u8\sumo_export"

# 車輛類型定義
VEHICLE_TYPES = {
    "car": {"accel": "2.6", "decel": "4.5", "sigma": "0.5", "length": "5.0", "maxSpeed": "70"},
    "bus": {"accel": "1.5", "decel": "3.0", "sigma": "0.5", "length": "12.0", "maxSpeed": "50"},
    "truck": {"accel": "1.0", "decel": "3.0", "sigma": "0.5", "length": "8.0", "maxSpeed": "50"},
    "motorcycle": {"accel": "3.0", "decel": "6.0", "sigma": "0.5", "length": "2.2", "maxSpeed": "80"},
}


def load_sumo_edge_map(net_file):
    """
    從 SUMO .net.xml 建立 OSM ID → SUMO edge ID 的映射。
    跳過 internal edges (以 ':' 開頭)。
    """
    print(f"讀取 SUMO 路網: {net_file}")
    tree = ET.parse(net_file)
    root = tree.getroot()
    edge_map = {}  # osm_way_id (int) → sumo_edge_id
    edge_info = {}  # sumo_edge_id → {type, lanes, speed, from, to}

    for edge in root.findall("edge"):
        eid = edge.get("id", "")
        # 跳過 internal edges
        if eid.startswith(":"):
            continue
        func = edge.get("function", "")
        if func == "internal":
            continue

        # 嘗試提取 OSM way ID (SUMO edge ID 通常是純數字的 OSM way ID)
        try:
            # 處理帶 suffix 的 edge ID (如 "1004040965#0")
            osm_id = int(eid.split("#")[0])
        except ValueError:
            continue

        edge_map[osm_id] = eid
        edge_info[eid] = {
            "type": edge.get("type", "").replace("highway.", ""),
            "priority": int(edge.get("priority", 1)),
            "from": edge.get("from", ""),
            "to": edge.get("to", ""),
        }
        # 讀取 lane 取得 speed
        lane = edge.find("lane")
        if lane is not None:
            edge_info[eid]["speed"] = float(lane.get("speed", "13.89"))
            edge_info[eid]["length"] = float(lane.get("length", "100"))

    print(f"  解析到 {len(edge_map)} 個有效 SUMO 邊 (排除 internal)")
    return edge_map, edge_info


def load_edge_flows(data_dir, date_str=None):
    """
    讀取所有 edge_flow JSONL 檔案，聚合每條邊每小時的平均流量。
    回傳: {(osm_way_id, hour): {"avg_score": ..., "count": ..., "highway": ...}}
    """
    edge_data = defaultdict(lambda: {"scores": [], "highway": "", "lanes": "", "maxspeed": ""})

    if date_str:
        dirs = [os.path.join(data_dir, date_str)]
    else:
        dirs = sorted([
            os.path.join(data_dir, d) for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.startswith("edge_flow_") or not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(d, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # 提取 OSM way ID
                    osm_raw = rec.get("osmid", "")
                    try:
                        if "[" in osm_raw:
                            osm_id = int(re.findall(r"\d+", osm_raw)[0])
                        else:
                            osm_id = int(osm_raw)
                    except (ValueError, IndexError):
                        continue

                    hour = rec.get("hour", 0)
                    score = rec.get("score", 0)
                    key = (osm_id, hour)
                    edge_data[key]["scores"].append(score)
                    if not edge_data[key]["highway"]:
                        edge_data[key]["highway"] = rec.get("highway", "")
                        edge_data[key]["lanes"] = str(rec.get("lanes", ""))
                        edge_data[key]["maxspeed"] = str(rec.get("maxspeed", ""))

    print(f"讀取到 {len(edge_data)} 個 (OSM ID, hour) 組合")
    return edge_data


def aggregate_to_flows(edge_data, edge_map, edge_info, hours=None):
    """
    將邊數據聚合為 SUMO flow 定義。
    回傳: list of flow dicts
    """
    flows = []
    matched = 0
    unmatched = 0

    for (osm_id, hour), data in sorted(edge_data.items()):
        if hours and hour not in hours:
            continue

        # 匹配 SUMO edge
        sumo_eid = edge_map.get(osm_id)
        if sumo_eid is None:
            # 嘗試模糊匹配 (OSM 節點可能對應 #0 suffix)
            for suffix in ["", "#0", "#1"]:
                test_id = f"{osm_id}{suffix}"
                if test_id in edge_info:
                    sumo_eid = test_id
                    break

        if sumo_eid is None:
            unmatched += 1
            continue
        matched += 1

        scores = data["scores"]
        avg_score = sum(scores) / len(scores) if scores else 0
        # 壅塞分數 → 車流量 (粗略估算)
        # score 0.3 = 暢通 (~300 veh/h), score 0.75 = 壅塞 (~800 veh/h)
        base_vph = 200 + avg_score * 800  # 每小時車輛數
        vph = int(min(1500, max(50, base_vph)))

        # 跳過極低流量邊
        if vph < 50 and avg_score < 0.05:
            continue

        flows.append({
            "edge_id": sumo_eid,
            "hour": hour,
            "avg_score": round(avg_score, 3),
            "vph": vph,
            "count": len(scores),
            "highway": data["highway"],
        })

    print(f"  匹配成功: {matched}, 失敗: {unmatched}")
    return flows


def generate_rou_xml(flows, output_path, hours=None):
    """生成 SUMO .rou.xml 檔案"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!-- Generated by convert_to_sumo.py -->\n')
        f.write(f'<!-- {len(flows)} flows from {len(set(f["edge_id"] for f in flows))} edges -->\n')
        f.write('<routes>\n\n')

        # 車輛類型
        for vtype_id, params in VEHICLE_TYPES.items():
            attrs = " ".join(f'{k}="{v}"' for k, v in params.items())
            f.write(f'    <vType id="{vtype_id}" {attrs}/>\n')
        f.write("\n")

        # 按小時分組
        hours_sorted = sorted(set(f["hour"] for f in flows))
        for hour in hours_sorted:
            hour_flows = [f for f in flows if f["hour"] == hour]
            begin = hour * 3600
            end = (hour + 1) * 3600
            f.write(f'    <!-- Hour {hour:02d}:00–{hour + 1:02d}:00 ({len(hour_flows)} flows) -->\n')
            for fl in hour_flows:
                # 使用 edge ID 作為 route (單一邊段的 route)
                route_id = f"r_{fl['edge_id']}_{hour:02d}"
                f.write(f'    <route id="{route_id}" edges="{fl["edge_id"]}"/>\n')
                flow_id = f"f_{fl['edge_id']}_{hour:02d}"
                f.write(
                    f'    <flow id="{flow_id}" type="car" route="{route_id}" '
                    f'begin="{begin}" end="{end}" vehsPerHour="{fl["vph"]}" '
                    f'color="1,1,0"/>\n'
                )
            f.write("\n")

        f.write('</routes>\n')

    print(f"輸出: {output_path}")


def generate_taz_xml(flows, edge_info, output_path):
    """
    生成 SUMO TAZ (Traffic Assignment Zone) 定義檔。
    將相近的 edge 分組為 TAZ，建立 OD 矩陣框架。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 按 highway 類型分組邊為 TAZ
    taz_groups = defaultdict(list)
    for fl in flows:
        hw = fl.get("highway", "unclassified")
        # 簡化分類
        if hw in ("motorway", "motorway_link", "trunk", "trunk_link"):
            cat = "highway"
        elif hw in ("primary", "primary_link", "secondary", "secondary_link"):
            cat = "arterial"
        else:
            cat = "local"
        taz_groups[cat].append(fl["edge_id"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<additional>\n')
        for cat, edges in taz_groups.items():
            edges_str = " ".join(edges[:50])  # 限制每個 TAZ 最多 50 條邊
            taz_id = f"taz_{cat}"
            f.write(f'    <taz id="{taz_id}" edges="{edges_str}"/>\n')
        f.write('</additional>\n')

    print(f"TAZ 輸出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert collected traffic data to SUMO routes")
    parser.add_argument("--date", type=str, default=None, help="Date string, e.g. 2026-05-28")
    parser.add_argument("--hours", type=str, default=None, help="Hour ranges, e.g. 7-9,17-19")
    parser.add_argument("--data-dir", type=str, default=TRAINING_DATA_DIR)
    parser.add_argument("--net-file", type=str, default=SUMO_NET_FILE)
    parser.add_argument("--output-dir", type=str, default=SUMO_OUTPUT_DIR)
    args = parser.parse_args()

    # 解析小時範圍
    hours_set = None
    if args.hours:
        hours_set = set()
        for part in args.hours.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                hours_set.update(range(int(lo), int(hi) + 1))
            else:
                hours_set.add(int(part))

    # 讀取 SUMO 邊映射
    edge_map, edge_info = load_sumo_edge_map(args.net_file)

    # 讀取流量數據
    edge_data = load_edge_flows(args.data_dir, args.date)
    if not edge_data:
        print("沒有找到流量數據！請先執行 v2_detector.py 收集數據。")
        sys.exit(1)

    # 聚合為 flows
    flows = aggregate_to_flows(edge_data, edge_map, edge_info, hours_set)
    if not flows:
        print("沒有匹配到任何 SUMO 邊！")
        sys.exit(1)

    # 統計
    total_edges = len(set(f["edge_id"] for f in flows))
    total_vph = sum(f["vph"] for f in flows)
    print(f"產生 {len(flows)} 個 flows, "
          f"{total_edges} 個不同邊, "
          f"總流量 {total_vph} veh/h")

    # 生成輸出
    date_tag = args.date or "all"
    rou_path = os.path.join(args.output_dir, f"macao_traffic_{date_tag}.rou.xml")
    generate_rou_xml(flows, rou_path, hours_set)

    taz_path = os.path.join(args.output_dir, f"macao_taz_{date_tag}.add.xml")
    generate_taz_xml(flows, edge_info, taz_path)

    # 生成使用說明
    readme_path = os.path.join(args.output_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("SUMO 車流檔案使用說明\n")
        f.write("====================\n\n")
        f.write("1. 將 .rou.xml 放到 SUMO config 目錄下\n")
        f.write("2. 在 .sumocfg 中加入:\n")
        f.write(f'   <route-files value="macao_traffic_{date_tag}.rou.xml"/>\n')
        f.write("3. (可選) 在 .sumocfg 中加入 TAZ:\n")
        f.write(f'   <additional-files value="macao_taz_{date_tag}.add.xml"/>\n')
        f.write("\n注意: flows 僅覆蓋有攝影機數據覆蓋的路段。\n")
        f.write("無數據路段需用 od2trips + duarouter 補齊。\n")

    print(f"\n完成！輸出目錄: {args.output_dir}")
    print(f"  {rou_path}")
    print(f"  {taz_path}")
    print(f"  {readme_path}")


if __name__ == "__main__":
    main()
