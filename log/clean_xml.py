import xml.etree.ElementTree as ET
import os

def clean_zero_flows(input_path, output_path):
    print(f"🔍 正在讀取檔案：{input_path} ...")
    
    try:
        # 1. 解析 XML 檔案，建立樹狀結構
        tree = ET.parse(input_path)
        root = tree.getroot()
        
        removed_count = 0
        
        # 2. 尋找並移除 number="0" 或 probability="0" 的項目
        # 在 ElementTree 中，我們需要透過父節點 (root) 來移除子節點
        # 使用 list(root) 是為了避免在迴圈中修改長度導致報錯
        for child in list(root):
            # 通常車流會用 <flow> 或 <vehicle> 標籤
            if child.tag in ['flow', 'vehicle']:
                # 取得 number 或 probability 的屬性值
                num_value = child.get('number')
                prob_value = child.get('probability')
                
                # 如果車輛數為 0，或者機率為 0，就把它刪除
                if num_value == '0' or prob_value in ['0', '0.0']:
                    root.remove(child)
                    removed_count += 1
                    
        # 3. 儲存成新的乾淨檔案
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        print("\n🎉 清理完成！")
        print(f"🗑️ 共刪除了 {removed_count} 筆數值為 0 的無效車流。")
        print(f"💾 乾淨的檔案已儲存為：{output_path}")
        
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到檔案 '{input_path}'，請檢查路徑是否正確。")
    except ET.ParseError:
        print("❌ 錯誤：XML 檔案格式損壞，無法解析。")
    except Exception as e:
        print(f"❌ 發生未知的錯誤：{e}")

# ==========================================
# 執行區塊：請把下面的檔名換成你實際的檔案路徑
# ==========================================
if __name__ == "__main__":
    # 你的原始 .rou.xml 檔案路徑
    original_file = r"D:\school\bsd\log\macao.rou.xml"
    
    # 清理後要存成的新檔案路徑 (加個 _cleaned 避免搞混)
    cleaned_file = r"D:\school\bsd\log\macao_cleaned.rou.xml"
    
    clean_zero_flows(original_file, cleaned_file)