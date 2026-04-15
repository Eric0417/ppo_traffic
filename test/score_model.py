import numpy as np
from stable_baselines3 import PPO
import traci
from sumo_env import MacauSumoEnv

def evaluate_and_score(model_path, net_file, route_file, steps=1800):
    print("==================================================")
    is_baseline = (model_path is None)
    
    if is_baseline:
        print(f"🚦 啟動 Baseline 基準評分程序 (不使用 AI，測試步數: {steps})")
    else:
        print(f"🚦 啟動 AI 模型評分程序 (測試步數: {steps})")
    print("==================================================")

    # 1. 初始化環境
    env = MacauSumoEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        min_green=15,
        max_steps=steps,
        # scale=0.3  # 如果你的 sumo_env 沒吃到這個參數可以先註解掉
    )

    # 2. 載入模型 (僅當非 Baseline 時)
    model = None
    if not is_baseline:
        try:
            model = PPO.load(model_path, device='cuda')
            print("[+] 模型載入成功！開始測驗...\n")
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return

    # 計分板變數
    total_reward = 0.0
    total_queue_sum = 0
    total_wait_time_sum = 0
    total_arrived = 0  # 🌟 新增：用來累計所有步驟中成功抵達的車輛
    
    obs, _ = env.reset()

    # 3. 開始執行模擬
    for step in range(steps):
        
        if is_baseline:
            # 【Baseline 模式】: AI 不介入，全輸出 0 讓 SUMO 跑預設的定時紅綠燈
            action = np.zeros(env.action_space.shape[0], dtype=int)
            
        else:
            # 【AI 模式】: 動態獲取模型需要的維度
            expected_obs_len = model.observation_space.shape[0]
            expected_act_len = env.action_space.shape[0]

            # --- 眼睛補齊 (Observation Padding) ---
            current_obs = obs
            if len(current_obs) < expected_obs_len:
                current_obs = np.concatenate([current_obs, np.zeros(expected_obs_len - len(current_obs))])
            else:
                current_obs = current_obs[:expected_obs_len]

            # AI 思考動作
            action, _ = model.predict(current_obs, deterministic=True)

            # --- 手部補齊 (Action Padding) ---
            if len(action) < expected_act_len:
                action = np.concatenate([action, np.zeros(expected_act_len - len(action), dtype=action.dtype)])
            else:
                action = action[:expected_act_len]

        # 與環境互動
        obs, reward, terminated, truncated, info = env.step(action)

        # 收集數據
        total_reward += reward
        total_queue_sum += info.get("total_queue", 0)
        
        # 🌟 關鍵修正：每一秒鐘都把成功壓線的車子加進總數裡
        total_arrived += traci.simulation.getArrivedNumber()
        
        # 🌟 效能優化：直接從 info 拿等待時間，避免大量車輛造成的卡頓
        total_wait_time_sum += info.get("total_wait", 0)

        if (step + 1) % 300 == 0:
            print(f"  ➜ 測驗進度: {step + 1}/{steps} 步")

        if terminated or truncated:
            break

    # 4. 取得最終結算數據
    final_throughput = total_arrived  # 🌟 直接使用累加的總和
    avg_queue = total_queue_sum / steps
    avg_wait = total_wait_time_sum / steps if final_throughput > 0 else 0

    env.close()

    # =================================================================
    # 🏆 計算綜合評分
    # =================================================================
    base_score = (final_throughput * 20) - (avg_queue * 2) + (total_reward * 0.01)
    final_score = max(0, int(base_score)) 

    # 5. 印出成績單
    print("\n==================================================")
    print("📝 【交通號誌控制 - 最終成績單】")
    print("==================================================")
    
    # 根據是否為 Baseline 顯示不同的測試名稱
    model_name_display = "Baseline (SUMO 原生定時號誌)" if is_baseline else model_path.split(chr(92))[-1]
    print(f"🔹 測試模型: {model_name_display}")
    print(f"🔹 總測試時間: {steps} 秒")
    print("--------------------------------------------------")
    print(f"✅ 成功疏導車輛 (Throughput): {final_throughput} 輛")
    print(f"⚠️ 平均排隊長度 (Avg Queue):  {avg_queue:.2f} 輛/秒")
    print(f"⏱️ 系統總體獎勵 (Total Reward): {total_reward:.2f}")
    print("--------------------------------------------------")
    print(f"🏆 綜合交通評分 (Traffic Score): 【 {final_score} 分 】")
    print("==================================================")


if __name__ == "__main__":
    NET_FILE = r"D:\school\bsd\test\macao.net.xml"
    ROUTE_FILE = r"D:\school\bsd\log\macao_perfect.rou.xml"
    MODEL_PATH = r"D:\school\bsd\v5_result\ppo_macao_fast_260000_steps.zip"
    BL = None
    # 傳入 None 代表不使用 AI，直接跑 Baseline
    evaluate_and_score(MODEL_PATH, NET_FILE, ROUTE_FILE, steps=2400)

'''

==================================================
📝 【交通號誌控制 - 最終成績單】
==================================================
🔹 測試模型: Baseline (SUMO 原生定時號誌)
🔹 總測試時間: 2400 秒
--------------------------------------------------
✅ 成功疏導車輛 (Throughput): 1261 輛
⚠️ 平均排隊長度 (Avg Queue):  40.23 輛/秒
⏱️ 系統總體獎勵 (Total Reward): -12122.02
--------------------------------------------------
🏆 綜合交通評分 (Traffic Score): 【 25018 分 】
==================================================

==================================================
📝 【交通號誌控制 - 最終成績單】
==================================================
🔹 測試模型: ppo_macao_fast_40000_steps.zip
🔹 總測試時間: 2400 秒
--------------------------------------------------
✅ 成功疏導車輛 (Throughput): 1253 輛
⚠️ 平均排隊長度 (Avg Queue):  38.32 輛/秒
⏱️ 系統總體獎勵 (Total Reward): -12723.34
--------------------------------------------------
🏆 綜合交通評分 (Traffic Score): 【 24856 分 】
==================================================

==================================================
📝 【交通號誌控制 - 最終成績單】
==================================================
🔹 測試模型: ppo_macao_fast_80000_steps.zip
🔹 總測試時間: 2400 秒
--------------------------------------------------
✅ 成功疏導車輛 (Throughput): 1273 輛
⚠️ 平均排隊長度 (Avg Queue):  38.42 輛/秒
⏱️ 系統總體獎勵 (Total Reward): -10074.56
--------------------------------------------------
🏆 綜合交通評分 (Traffic Score): 【 25282 分 】
==================================================

==================================================
📝 【交通號誌控制 - 最終成績單】
==================================================
🔹 測試模型: ppo_macao_fast_100000_steps.zip
🔹 總測試時間: 2400 秒
--------------------------------------------------
✅ 成功疏導車輛 (Throughput): 1272 輛
⚠️ 平均排隊長度 (Avg Queue):  36.17 輛/秒
⏱️ 系統總體獎勵 (Total Reward): -9327.64
--------------------------------------------------
🏆 綜合交通評分 (Traffic Score): 【 25274 分 】
==================================================

==================================================
📝 【交通號誌控制 - 最終成績單】
==================================================
🔹 測試模型: ppo_macao_fast_260000_steps.zip
🔹 總測試時間: 2400 秒
--------------------------------------------------
✅ 成功疏導車輛 (Throughput): 1279 輛
⚠️ 平均排隊長度 (Avg Queue):  38.26 輛/秒
⏱️ 系統總體獎勵 (Total Reward): -10442.61
--------------------------------------------------
🏆 綜合交通評分 (Traffic Score): 【 25399 分 】
==================================================

'''