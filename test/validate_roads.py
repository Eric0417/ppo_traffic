import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sumo_env import MacauSumoEnv

def validate_model(model_path, net_file, route_file, steps_to_run=1000):
    print("==================================================")
    print("啟動澳門全局交通大腦驗證程序 (Sim2Real Validation)")
    print("==================================================")
    
    # 1. 初始化環境 (開啟 GUI 讓評判能看見視覺效果)
    env = MacauSumoEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=True,      # 開啟 SUMO-GUI
        min_green=15,      # 設定硬體防呆限制
        max_steps=steps_to_run
    )
    
    # 2. 載入訓練好的 PPO 模型
    print(f"[*] 正在載入模型: {model_path} ...")
    try:
        model = PPO.load(model_path)
        print("[+] 模型載入成功！")
    except Exception as e:
        print(f"[-] 模型載入失敗: {e}")
        print("[!] 將使用隨機動作進行 Baseline 測試以供對比。")
        model = None

    obs, _ = env.reset()
    
    queue_history = []
    reward_history = []
    
    print(f"[*] 開始模擬驗證，總步數: {steps_to_run} 步...")
    
    # 3. 執行模擬迴圈
    for step in range(steps_to_run):
        if model is not None:
            # 讓 PPO 根據當前狀態預測下一步燈號動作
            # deterministic=True 代表使用最佳決策，不進行隨機探索
            action, _states = model.predict(obs, deterministic=True)
        else:
            # 如果沒有模型，做隨機測試 (Baseline)
            action = env.action_space.sample()
            
        # 將動作輸入環境
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 紀錄數據用於畫圖
        queue_history.append(info["total_queue"])
        reward_history.append(reward)
        
        # 每 100 步在終端機印出進度
        if (step + 1) % 100 == 0:
            print(f"進度: {step + 1}/{steps_to_run} 步 | "
                  f"當前全局排隊總數: {info['total_queue']} 輛 | "
                  f"即時獎勵 (懲罰值): {reward:.2f}")
            
        if terminated or truncated:
            break

    print("[+] 模擬驗證完成！正在關閉環境...")
    env.close()
    
    # 4. 生成效能驗證報告與圖表 (給評判看的鐵證)
    print("==================================================")
    print("效能驗證報告 (Validation Report):")
    print(f"平均全局排隊車輛: {np.mean(queue_history):.2f} 輛")
    print(f"最大排隊峰值: {np.max(queue_history)} 輛")
    print("==================================================")
    
    plot_results(queue_history)

def plot_results(queue_history):
    """繪製排隊長度趨勢圖"""
    plt.figure(figsize=(10, 5))
    plt.plot(queue_history, label="Total Queuing Vehicles", color='#d62728', linewidth=2)
    plt.title("Macau 1860 Intersections: AI Control Validation", fontsize=14, fontweight='bold')
    plt.xlabel("Simulation Steps (Seconds)", fontsize=12)
    plt.ylabel("Number of Waiting Vehicles", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 儲存圖片並顯示
    plt.savefig("validation_result.png", dpi=300)
    print("[*] 圖表已儲存為 validation_result.png")
    plt.show()

if __name__ == "__main__":
    # 使用你提供的實際檔案路徑
    MODEL_PATH = r"D:\school\bsd\models\ppo_macao_v3_425000_steps.zip"  
    NET_FILE = r"D:\school\bsd\log\macao.net.xml"          
    ROUTE_FILE = r"D:\school\bsd\log\macao_hell.rou.xml"        
    
    validate_model(MODEL_PATH, NET_FILE, ROUTE_FILE, steps_to_run=1000)