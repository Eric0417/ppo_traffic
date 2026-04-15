import sys
import time

# =====================================================================
# 【神級外掛：極限順序版】
# 必須在載入任何科學運算庫之前完成偽裝，避免底層 C 擴展直接抓到錯誤路徑
try:
    import numpy as np
    # 強行注入偽裝
    if 'numpy.core' in sys.modules:
        sys.modules['numpy._core'] = sys.modules['numpy.core']
    if 'numpy.core.numeric' in sys.modules:
        sys.modules['numpy._core.numeric'] = sys.modules['numpy.core.numeric']
except ImportError:
    print("無法載入 NumPy，請確認虛擬環境已安裝 numpy<2.0")
# =====================================================================

import matplotlib
# 如果在沒有螢幕的環境跑，請取消註解下面這行
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
import traci
from sumo_env import MacauSumoEnv

# ... [其餘 plot_multiple_results 函數保持不變] ...

def run_simulation(model_path, net_file, route_file, steps, use_gui=False, label=""):
    """
    執行單次模擬並收集評估數據
    """
    print(f"[-] 正在初始化環境 ({label})...")
    
    # 建立環境
    try:
        env = MacauSumoEnv(
            net_file=net_file,
            route_file=route_file,
            use_gui=use_gui,
            min_green=15,
            max_steps=steps
        )
    except Exception as e:
        print(f"[!] 環境初始化失敗: {e}")
        return None

    # 載入模型
    model = None
    if model_path is not None:
        print(f"[-] 正在載入模型: {model_path}")
        try:
            # 強制指定 device='cpu' 通常可以避開一些 GPU/NumPy 衝突
            model = PPO.load(model_path, device='cpu') 
            print("[+] 模型載入成功！")
        except Exception as e:
            print(f"[-] 模型載入失敗: {e}")
            print("[!] 將使用隨機動作進行 Baseline 測試。")

    time_steps, queue_lengths, wait_times, throughputs = [], [], [], []
    cumulative_arrived = 0

    obs, _ = env.reset()

    print(f"[-] 開始執行模擬 ({steps} 步)...")
    for step in range(steps):
        # 1. 處理狀態觀察值 (Observation Padding)
        current_obs = obs
        if model is not None:
            # 補齊到 667 維度
            if len(current_obs) < 667:
                current_obs = np.concatenate([current_obs, np.zeros(667 - len(current_obs))])
            else:
                current_obs = current_obs[:667]
            
            # 預測
            action, _ = model.predict(current_obs, deterministic=True)
            
            # 2. 處理動作指令 (Action Padding)
            expected_act = env.action_space.shape[0]
            if len(action) < expected_act:
                action = np.concatenate([action, np.zeros(expected_act - len(action))])
            else:
                action = action[:expected_act]
        else:
            action = env.action_space.sample()

        # 執行動作
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 數據收集
            current_halted = info.get("total_queue", 0)
            veh_ids = traci.vehicle.getIDList()
            current_wait_time = sum([traci.vehicle.getWaitingTime(v_id) for v_id in veh_ids])
            cumulative_arrived += traci.simulation.getArrivedNumber()
            
            time_steps.append(step)
            queue_lengths.append(current_halted)
            wait_times.append(current_wait_time)
            throughputs.append(cumulative_arrived)

        except Exception as e:
            print(f"[!] 模擬運行中斷: {e}")
            break

        if (step + 1) % 300 == 0:
            print(f"進度: {step + 1}/{steps} | 排隊: {current_halted} | 抵達: {cumulative_arrived}")

        if terminated or truncated:
            break

    env.close()
    print(f"[+] {label} 模擬完成！\n")
    return time_steps, queue_lengths, wait_times, throughputs

if __name__ == "__main__":
    NET_FILE = r"D:\school\bsd\test\macao.net.xml"
    ROUTE_FILE = r"D:\school\bsd\test\macao_real_hell_1.rou.xml"
    STEPS = 1800  # 測試 30 分鐘

    # 你的模型競技場清單
    MODELS_TO_TEST = {
        "Baseline (隨機動作)": None,
        "PPO_v3 (50,000 steps)": r"D:\school\bsd\models\ppo_macao_v3_50000_steps.zip",
        "PPO_v3 (425,000 steps)": r"D:\school\bsd\models\ppo_macao_v3_425000_steps.zip",
    }

    all_results = {}

    print("==================================================")
    print("啟動澳門全局交通大腦驗證程序 (Multi-Model Comparison)")
    print("==================================================")

    for label, model_path in MODELS_TO_TEST.items():
        print(f"\n========================================")
        print(f"[>>>] 開始測試: {label}")
        print(f"========================================")
        
        # 呼叫模擬函數，use_gui=False 可以讓腳本跑得飛快
        result_data = run_simulation(model_path, NET_FILE, ROUTE_FILE, STEPS, use_gui=False, label=label)
        
        all_results[label] = result_data

    # 畫出最終的三合一圖表
    plot_multiple_results(all_results)