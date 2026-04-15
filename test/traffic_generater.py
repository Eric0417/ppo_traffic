import os
import sys
import random
import sumolib

def generate_macao_route_file(net_file, output_file, steps=3600):
    print("==================================================")
    print(f"[*] 正在讀取澳門路網: {net_file} ... (這可能需要幾秒鐘)")
    
    try:
        net = sumolib.net.readNet(net_file)
        
        valid_edge_objs = []
        for e in net.getEdges():
            if e.isSpecial():
                continue
            lanes = e.getLanes()
            # 只要能走車，就先列入候選清單
            if any(lane.allows("passenger") for lane in lanes):
                valid_edge_objs.append(e)
                
        print(f"[+] 成功載入路網，共找到 {len(valid_edge_objs)} 條合法街道。")
        
    except Exception as e:
        print(f"[-] 讀取路網失敗: {e}")
        return

    print(f"[*] 正在生成車流檔案: {output_file} ... (正在計算連通路徑，請稍候)")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n\n')
        
        f.write('    \n')
        f.write('    <vType id="scooter" vClass="motorcycle" length="2.5" width="1.0" maxSpeed="15.0" accel="3.5" decel="4.0" sigma="0.8" />\n')
        f.write('    <vType id="private_car" vClass="passenger" length="4.5" width="1.8" maxSpeed="20.0" accel="2.0" decel="2.5" sigma="0.5" />\n')
        f.write('    <vType id="taxi" vClass="taxi" length="4.5" width="1.8" maxSpeed="25.0" accel="2.5" decel="3.0" sigma="0.7" color="0,0,0"/>\n')
        f.write('    <vType id="casino_bus" vClass="coach" length="12.0" width="2.5" maxSpeed="15.0" accel="0.8" decel="1.5" sigma="0.1" color="255,215,0"/>\n\n')

        vehicle_types = ["scooter", "private_car", "taxi", "casino_bus"]
        type_weights = [0.40, 0.35, 0.15, 0.10] 
        
        # 對應的 SUMO vClass (用於導航檢查)
        vclass_map = {
            "scooter": "motorcycle", 
            "private_car": "passenger", 
            "taxi": "taxi", 
            "casino_bus": "coach"
        }
        
        vehicle_id_counter = 0
        
        for step in range(steps):
            base_prob = 0.5 + 0.3 * random.uniform(-1, 1) 
            if (step % 300) < 60: 
                base_prob += 0.8 

            if random.random() < base_prob:
                num_cars_this_second = random.randint(1, 5)
                
                for _ in range(num_cars_this_second):
                    v_type = random.choices(vehicle_types, weights=type_weights)[0]
                    v_class = vclass_map[v_type]
                    
                    # [★ 路徑連通性保證機制 ★]
                    # 最多嘗試 15 次，確保找到「真的有路可以通」的起終點
                    valid_path_found = False
                    for _attempt in range(15):
                        src_edge = random.choice(valid_edge_objs)
                        dst_edge = random.choice(valid_edge_objs)
                        
                        if src_edge == dst_edge:
                            continue
                            
                        # 呼叫 SUMO 底層導航引擎，檢查這兩條路之間是否有路徑
                        path, cost = net.getShortestPath(src_edge, dst_edge, vClass=v_class)
                        
                        if path is not None: # 如果 path 不是 None，代表找得到路！
                            valid_path_found = True
                            break
                            
                    # 只有確定有路，才寫入 XML 檔案
                    if valid_path_found:
                        f.write(f'    <trip id="v_{vehicle_id_counter}" type="{v_type}" depart="{step}.00" from="{src_edge.getID()}" to="{dst_edge.getID()}"/>\n')
                        vehicle_id_counter += 1

        f.write('</routes>\n')
    print(f"[+] 生成完畢！共生成了 {vehicle_id_counter} 輛【保證有路線可走】的車。")
    print("==================================================")

if __name__ == "__main__":
    NET_FILE = r"D:\school\bsd\test\macao.net.xml"
    OUTPUT_ROU = r"D:\school\bsd\test\macao_real_hell_1.rou.xml"
    
    # 生成 1800 秒 (30 分鐘) 的模擬數據
    generate_macao_route_file(NET_FILE, OUTPUT_ROU, steps=1800)