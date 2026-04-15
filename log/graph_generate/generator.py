import pandas as pd
import folium
from folium.plugins import HeatMapWithTime
import math
import os

# =========================
# 1. 街道座標設定 (與你提供的一致)
# =========================
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

# =========================
# 2. 空間距離與網路拓撲計算
# =========================
def haversine_distance(coord1, coord2):
    """計算兩點經緯度之間的距離（公尺）"""
    R = 6371000  # 地球半徑 (公尺)
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def build_neighbor_graph(coords, threshold_meters=400):
    """建立道路相鄰圖，距離小於 threshold_meters 視為鄰居 (N_i)"""
    neighbors = {node: [] for node in coords}
    for node1, coord1 in coords.items():
        for node2, coord2 in coords.items():
            if node1 != node2:
                dist = haversine_distance(coord1, coord2)
                if dist <= threshold_meters:
                    neighbors[node1].append(node2)
    return neighbors

# =========================
# 3. 實作你提供的熱傳導方程式
# =========================
def apply_heat_conduction(data_row, neighbors, beta=0.3):
    """
    套用公式： x_i^(t+1) = (1 - \beta)x_i^(t) + \beta( 1/|N_i| \sum x_j^(t) )
    data_row: 特定時間的塞車數值 (Pandas Series)
    beta: 熱傳導係數 (預設 0.3，代表 30% 受到周圍鄰居影響)
    """
    smoothed_row = data_row.copy()
    
    for node in data_row.index:
        # 確保該街道有座標紀錄，且目前數值不是空值
        if node in neighbors and pd.notna(data_row[node]):
            node_neighbors = neighbors[node]
            
            # 取得所有有效鄰居的當下塞車值 x_j^(t)
            neighbor_vals = [data_row[n] for n in node_neighbors 
                             if n in data_row.index and pd.notna(data_row[n])]
            
            if len(neighbor_vals) > 0:
                # 計算鄰居的平均值： 1/|N_i| \sum x_j^(t)
                neighbor_mean = sum(neighbor_vals) / len(neighbor_vals)
                
                # 執行熱傳導公式
                smoothed_val = (1 - beta) * data_row[node] + (beta * neighbor_mean)
                smoothed_row[node] = smoothed_val
                
    return smoothed_row

# =========================
# 4. 主程式生成熱力圖
# =========================
def generate_smoothed_timeline_heatmap(csv_file_path, output_file="macau_heat_conduction_timeline.html"):
    print(f"正在讀取資料: {csv_file_path} ...")
    
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
    time_col = df.columns[0]
    street_cols = df.columns[1:]
    
    # 清理資料：刪除全為 0 的行
    df = df[df[street_cols].sum(axis=1) > 0]
    
    # 處理時間並計算原始歷史平均
    df[time_col] = pd.to_datetime(df[time_col])
    df['Hour'] = df[time_col].dt.hour
    hourly_avg = df.groupby('Hour')[street_cols].mean()
    
    # 建立道路網路拓撲 (400 公尺為鄰居界線，可自行微調)
    print("正在建立道路拓撲網路 (計算 N_i)...")
    neighbors_graph = build_neighbor_graph(ORIGINAL_COORDS, threshold_meters=400)
    
    heat_data = []
    time_index = []
    
    print("正在套用圖熱傳導方程式 (Graph Heat Equation)...")
    for hour in range(24):
        time_index.append(f"{hour:02d}:00")
        hour_data = []
        
        if hour in hourly_avg.index:
            raw_data = hourly_avg.loc[hour]
            
            # === 在這裡套用你的數學公式 ===
            # 將原始數據進行一次熱傳導平滑，模擬塞車向周圍擴散
            smoothed_data = apply_heat_conduction(raw_data, neighbors_graph, beta=0.35)
            
            for street_name, jam_value in smoothed_data.items():
                if street_name in ORIGINAL_COORDS and pd.notna(jam_value) and jam_value > 0:
                    lat, lon = ORIGINAL_COORDS[street_name]
                    hour_data.append([lat, lon, float(jam_value)])
                    
        heat_data.append(hour_data)
    
    # 建立地圖與時間軸插件
    MACAU_CENTER = [22.195, 113.545]
    m = folium.Map(location=MACAU_CENTER, zoom_start=14, tiles='CartoDB dark_matter')
    
    # 5. 加入動態時間軸熱力圖圖層 (調整為手動滑動模式)
    HeatMapWithTime(
        heat_data,
        index=time_index,         # 對應的時間標籤 (00:00 ~ 23:00)
        auto_play=False,          # 🔴 關閉自動播放，讓你可以自己滑動
        radius=28,                
        min_opacity=0.3,
        max_opacity=0.8,
        use_local_extrema=False,  
        display_index=True,       # 顯示目前的時間標籤
        # 可以透過 position 調整時間軸位置，預設為 'bottomleft' (左下角)
        position='bottomleft'     
    ).add_to(m)
    
    # 加入大標題與操作提示
    title_html = '''
         <h3 align="center" style="font-size:20px; color:white; margin-top: 10px;">
         <b>澳門路網 24 小時歷史塞車熱力圖</b><br>
         <span style="font-size:14px; color:gray;">💡 請使用左下角的時間軸自由滑動查看不同時間的塞車情況</span>
         </h3>
         '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # 6. 輸出最終 HTML 檔案
    m.save(output_file)
    print(f"✅ 成功生成手動滑動版熱力圖 -> {output_file}")

if __name__ == "__main__":
    # 將這裡替換成你的 CSV 檔案實際路徑
    CSV_FILE_PATH = r"D:\school\bsd\log\traffic_jam_log.csv" 
    OUTPUT_FILE = r"D:\school\bsd\log\graph_generate\macau_heat_conduction_timeline.html"
    
    generate_smoothed_timeline_heatmap(CSV_FILE_PATH, OUTPUT_FILE)