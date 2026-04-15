import traci
import time
from sumolib import checkBinary

try:
    sumoBinary = checkBinary('sumo') # 繼續用純文字版，跑得快又穩！
except Exception as e:
    print("錯誤：找不到 SUMO 執行檔。")
    exit()

config_path = r"D:\school\bsd\log\macao.sumocfg"
sumoCmd = [sumoBinary, "-c", config_path]

print(f"正在使用設定檔: {config_path}")
print("正在啟動 SUMO 並建立連線...")

try:
    traci.close()
except:
    pass

traci.start(sumoCmd)
tl_id = "cluster_1936845009_1944086504_3109979021_3109979027_#5more"

step = 0
try:
    # 這次我們讓模擬器跑 3000 步 (3000秒)，比較有機會看到車潮
    while step < 3000:
        traci.simulationStep()
        
        # 【💡 新增動作 (Action)：每 100 秒強制切換一次紅綠燈階段】
        if step % 100 == 0:
            # 取得目前的燈號階段 (Phase) 索引值
            current_phase_index = traci.trafficlight.getPhase(tl_id)
            
            # 強制把它切換到下一個階段 (+1)
            # 這就是未來你的 AI 決定 "我要改變燈號" 時會呼叫的指令！
            next_phase = current_phase_index + 1
            
            # 因為我們不知道它總共有幾個階段，如果超出範圍會報錯，所以用 try 保護一下
            try:
                traci.trafficlight.setPhase(tl_id, next_phase)
                print(f"⚡ [上帝之手介入] 時間 {step} 秒 | 已強制將燈號切換為下一個階段！")
            except traci.exceptions.TraCIException:
                # 如果到底了，就切回第 0 階段
                traci.trafficlight.setPhase(tl_id, 0)
                print(f"⚡ [上帝之手介入] 時間 {step} 秒 | 已強制將燈號重置回階段 0！")
                
        # 每 10 秒印出一次狀態
        if step % 10 == 0:
            current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            halting_vehicles = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in set(lanes)])
            
            # 為了讓終端機乾淨一點，我們設定「有車停等」或「上帝之手介入後」才印出特別提示
            if halting_vehicles > 0:
                print(f"⏳ 時間 {step} 秒 | 狀態: {current_state} | 🚗 發現塞車！停等車輛: {halting_vehicles} 輛")
            
        step += 1

except traci.exceptions.FatalTraCIError:
    print("模擬器已關閉或崩潰。")
except Exception as e:
    print(f"發生其他錯誤: {e}")
finally:
    traci.close()
    print("連線已安全結束。")