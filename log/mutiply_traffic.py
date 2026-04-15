import xml.etree.ElementTree as ET

def boost_traffic_smart(input_path, output_path, multiplier=5):
    print(f"🔥 正在將 {input_path} 的車流量聰明地放大 {multiplier} 倍...")
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()
        
        # 尋找所有的 flow (車流標籤)
        for flow in root.findall('flow'):
            
            # 狀況 A：如果原始檔案是用 vehsPerHour (每小時車流量)
            if 'vehsPerHour' in flow.attrib:
                current_vph = float(flow.get('vehsPerHour'))
                flow.set('vehsPerHour', str(current_vph * multiplier))
                
            # 狀況 B：如果原始檔案是用 number (固定總數)
            elif 'number' in flow.attrib:
                current_num = int(flow.get('number'))
                flow.set('number', str(int(current_num * multiplier)))
                
            # 狀況 C：如果原始檔案是用 period (每隔幾秒發出一輛車)
            elif 'period' in flow.attrib:
                current_period = float(flow.get('period'))
                # 時間間隔變短 = 車變多 (避免除以 0，最小設為 0.1 秒)
                new_period = max(0.1, current_period / multiplier)
                flow.set('period', str(new_period))
                
            # 狀況 D：如果原始檔案是用 probability (每秒生成的機率)
            elif 'probability' in flow.attrib:
                current_prob = float(flow.get('probability'))
                # 機率最高只能是 1.0 (100%)
                new_prob = min(1.0, current_prob * multiplier)
                flow.set('probability', str(new_prob))

        # 存檔
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"✅ 成功！無衝突的【地獄級車流檔】已儲存至：{output_path}")
        
    except Exception as e:
        print(f"❌ 發生錯誤：{e}")

# ==========================================
# 執行區塊
# ==========================================
if __name__ == "__main__":
    # ⚠️ 注意：這裡一定要讀取你「原本乾淨的」檔案，不要讀到剛剛壞掉的那個
    source_file = r"D:\school\bsd\log\macao_cleaned.rou.xml"
    
    # 輸出的地獄檔
    hell_file = r"D:\school\bsd\log\macao_hell.rou.xml"
    
    # 執行聰明放大 (這裡設定 5 倍)
    boost_traffic_smart(source_file, hell_file, multiplier=5)