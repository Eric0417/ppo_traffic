import sumolib
import csv

# 1. 讀取路網 (為了找出每個路口的進出道路)
net = sumolib.net.readNet(r"D:\school\bsd\SUMO\macao.net.xml")

# 2. 貼上你上一步跑出來成功的 mapping
mapping = {
    "提督馬路與高士德大馬路交界（向高士德大馬路方向）": "cluster_1936845009_1944086504_3109979021_3109979027_#5more",
    "巴坡沙大馬路與青洲大馬路交界": "1522474782",
    "提督馬路與罅些喇海軍上將巷交界": "cluster_10832091046_10832091047_11245156598_11245156599_#6more",
    "提督馬路與高士德大馬路交界": "cluster_1936845009_1944086504_3109979021_3109979027_#5more",
    "沙梨頭海邊街與爹美刁施拿地大馬路交界": "cluster_1809540489_1941289233_3660397476_4019772915_#5more",
    "勞動節大馬路": "3265331002",
    "慕拉士大馬路": "5206522788",
    "友誼橋大馬路(向氹仔方向)": "1433340230#1-AddedOnRampNode",
    "友誼大馬路(向馬揸度博士大馬路方向)": "1523899495",
    "馬揸度博士大馬路": "6646383974",
    "馬揸度博士大馬路(近勞工局)": "6562195264",
    "馬場北大馬路與馬場東大馬路交界": "1523899695",
    "友誼圓形地與友誼橋大馬路交界(向港珠澳大橋入口方向)": "5295497334",
    "美副將大馬路與俾利喇街交界": "cluster_1857073176_1936887950_4680466141_4680466142_#3more",
    "東北大馬路與黑沙環中街交界向友誼圓形地方向": "cluster_10197360195_1523899598",
    "黑沙環新街": "cluster_3377693019_6650037275_7516447556_7516447569_#1more",
    "馬交石斜坡與俾利喇街交界": "7874298255",
    "A2橋澳門出口": "1846195321",
    "友誼大馬路": "cluster_7756409798_7756409808",
    "馬六甲街停車場": "cluster_10209573966_10209573967_1942291904_2168717511",
    "松山隧道羅理基方向": "5108073970",
    "羅理基博士大馬路行車隧道往松山隧道方向": "1942779508",
    "宋玉生廣場": "7192410510",
    "捐血中心": "3534892108",
    "孫逸仙大馬路與城市日大馬路交界": "cluster_1942291743_1942291745_1942291748_2917006396_#7more",
    "孫逸仙大馬路(近終審法院前地)": "518464523",
    "西灣湖廣場(向孫逸仙大馬路)": "1420845480",
    "西灣湖廣場(向西灣湖景大馬路)": "1420845476",
    "高士德與俾利喇街交界": "cluster_1686120959_1938088908_1944086454_1944086459_#6more",
    "華士古停車場": "4076186062",
    "美副將大馬路與連勝馬路交界": "1832146663",
    "松山隧道高士德方向": "1449241548",
    "美副將大馬路與士多鳥拜斯大馬路交界": "cluster_1684887332_1832026537_1832026636_1938088901_#12more",
    "高偉樂街與荷蘭園大馬路交界": "cluster_10293199566_10293199569_1449239318_1725757020",
    "水坑尾": "4277845906",
    "沙梨頭海邊街": "4336638162",
    "亞馬喇前地之圓形地方向": "7026422036",
    "亞馬喇前地": "3007408120",
    "亞馬喇前地巴士站": "3007408120",
    "南灣大馬路與區華利前地交界": "1846784979",
    "南灣大馬路與殷皇子大馬路(向八角亭方向)": "6710502282",
    "南灣大馬路與殷皇子大馬路(向殷皇子大馬路方向)": "cluster_1791209584_1793358263_2027471666_2027471667_#9more",
    "巴素打爾古街近栢港停車場出口(向火船頭街方向)": "2168717691",
    "新馬路(近議事亭前地)向南灣大馬路方向": "cluster_1791209584_1793358263_2027471666_2027471667_#9more",
    "殷皇子大馬路與約翰四世大馬路交界": "13619373461",
    "殷皇子大馬路與葡京路交界": "3007408121",
    "水坑尾街與南灣大馬路交界路口": "3005526083",
    "沙梨頭海邊街與林茂巷交界": "cluster_1809540489_1941289233_3660397476_4019772915_#5more",
    "沙梨頭海邊街與魚鰓巷交界(向新馬路方向)": "1941289222",
    "爹美刁斯拿地大馬路近栢港停車場": "4016054596",
    "爹美刁施拿地大馬路與魚鱗巷交界(向十六浦方向)": "2629540340",
    "新馬路與巴素打爾古街交界": "4005738730",
    "新馬路與南灣大馬路交界": "cluster_1791209584_1793358263_2027471666_2027471667_#9more",
    "比厘喇馬忌士街與貨倉巷交界(向十六浦方向)": "2561573398",
    "比厘喇馬忌士街與馬博士巷交界": "4056323693",
    "河邊新街與比厘喇馬忌士街交界(向媽閣廟方向)": "3100110230",
    "河邊新街與航海學校街交界(向媽閣方向)": "4100191960",
    "河邊新街與鹽巷交界（向媽閣方向）": "cluster_2840210317_5543335568",
    "火船頭街近11號碼頭（向巴素打爾古街方向)": "4005738725",
    "火船頭街（近11號碼頭）（向河邊新街方向）": "7197553486"
}

# 🌟 這裡確保變數名稱正確
csv_filename = r"D:\school\bsd\log\traffic_jam_log.csv"
rou_filename = r"D:\school\bsd\log\macao.rou.xml"

print("開始生成 SUMO 車流文件...")

# 3. 打開並準備寫入 .rou.xml
with open(rou_filename, "w", encoding="utf-8") as f_out:
    # 寫入 XML 標頭，定義車輛類型
    f_out.write('<routes>\n')
    f_out.write('  <vType id="car" vClass="passenger" accel="2.6" decel="4.5" length="5.0" maxSpeed="15.0"/>\n')

    # 4. 讀取你的 CSV 數據 (🌟 這裡改成了 csv_filename 並換成 DictReader)
    with open(csv_filename, 'r', encoding='utf-8', errors='ignore') as f:
        # 使用 DictReader 才能透過字典 key (street_name) 去取值
        reader = csv.DictReader(line.replace('\0', '') for line in f)
        time_step = 0
        interval = 5  # 假設你的 CSV 每一行代表 5 分鐘 (300秒) 的間隔
        
        for row in reader:
            begin_time = time_step * interval
            end_time = begin_time + interval
            
            # 計數器
            flow_counter = 0 
            
            for street_name, sumo_id in mapping.items():
                # 確保 CSV 中真的有這個標題，且該格內容不是空白的
                if street_name in row and row[street_name]:
                    try:
                        # 🌟 防呆：清理字串中的百分比符號或空白，再轉換成浮點數
                        clean_value = row[street_name].replace('%', '').strip()
                        congestion = float(clean_value)
                    except ValueError:
                        continue 
                    
                    veh_per_hour = int(100 + (congestion * 1700))
                    
                    # 透過 SUMO ID 找到該路口
                    try:
                        node = net.getNode(sumo_id)
                    except KeyError:
                        continue 
                        
                    # 判斷函式，檢查這條路是否允許小客車行駛
                    def allows_passenger(edge):
                        for lane in edge.getLanes():
                            if lane.allows("passenger"):
                                return True
                        return False
                        
                    # 過濾出允許小客車進入和離開的道路
                    in_edges = [e for e in node.getIncoming() if allows_passenger(e)]
                    out_edges = [e for e in node.getOutgoing() if allows_passenger(e)]
                    
                    # 必須要有「允許汽車」的進跟出的路，才能產生車流
                    if in_edges and out_edges:
                        from_edge = in_edges[0].getID()
                        to_edge = out_edges[0].getID()
                        
                        flow_counter += 1
                        
                        flow_id = f"flow_{sumo_id}_{time_step}_{flow_counter}"
                        
                        f_out.write(f'  <flow id="{flow_id}" type="car" begin="{begin_time}" end="{end_time}" '
                                    f'from="{from_edge}" to="{to_edge}" vehsPerHour="{veh_per_hour}" departLane="best"/>\n')
            
            time_step += 1

    f_out.write('</routes>\n')

print(f"✅ 完成！已成功生成交通流檔案：{rou_filename}")