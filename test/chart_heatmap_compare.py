import numpy as np
from stable_baselines3 import PPO
import traci
from sumo_env import MacauSumoEnv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import concurrent.futures
import time

def run_evaluation(model_name, model_path, net_file, route_file, steps=1800):
    """
    執行單一模型的模擬測試 (將會在獨立的進程中運行)
    """
    is_baseline = (model_path is None)
    print(f"🚀 [啟動] {model_name} 正在載入環境...")

    # 1. 初始化環境
    try:
        env = MacauSumoEnv(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            min_green=15,
            max_steps=steps
        )
    except Exception as e:
        print(f"❌ {model_name} 環境啟動失敗: {e}")
        return None, None

    # 2. 載入模型
    model = None
    if not is_baseline:
        try:
            model = PPO.load(model_path, device='cuda')
        except Exception as e:
            print(f"❌ {model_name} 模型載入失敗: {e}")
            env.close()
            return None, None

    # 追蹤變數
    total_reward = 0.0
    total_queue_sum = 0
    total_wait_time_sum = 0
    total_arrived = 0 
    
    history = {'steps': [], 'arrived': [], 'queue': [], 'wait': [], 'reward': []}
    
    obs, _ = env.reset()

    # 3. 模擬迴圈
    for step in range(steps):
        if is_baseline:
            action = np.zeros(env.action_space.shape[0], dtype=int)
        else:
            expected_obs_len = model.observation_space.shape[0]
            expected_act_len = env.action_space.shape[0]

            current_obs = obs
            if len(current_obs) < expected_obs_len:
                current_obs = np.concatenate([current_obs, np.zeros(expected_obs_len - len(current_obs))])
            else:
                current_obs = current_obs[:expected_obs_len]

            action, _ = model.predict(current_obs, deterministic=True)

            if len(action) < expected_act_len:
                action = np.concatenate([action, np.zeros(expected_act_len - len(action), dtype=action.dtype)])
            else:
                action = action[:expected_act_len]

        obs, reward, terminated, truncated, info = env.step(action)

        current_queue = info.get("total_queue", 0)
        current_wait = info.get("total_wait", 0)
        total_arrived += traci.simulation.getArrivedNumber()
        
        total_reward += reward
        total_queue_sum += current_queue
        total_wait_time_sum += current_wait

        history['steps'].append(step)
        history['arrived'].append(total_arrived)
        history['queue'].append(current_queue)
        history['wait'].append(current_wait)
        history['reward'].append(reward)

        if terminated or truncated:
            break

    # 4. 結算
    final_throughput = total_arrived 
    avg_queue = total_queue_sum / steps
    env.close()

    base_score = (final_throughput * 15) - (avg_queue * 30) + (total_reward * 0.01)
    final_score = max(0, int(base_score)) 

    stats = {
        'Throughput': final_throughput,
        'Avg Queue': avg_queue,
        'Total Reward': total_reward,
        'Traffic Score': final_score
    }
    
    print(f"✅ [完成] {model_name} | 分數: {final_score} | 疏導量: {final_throughput}")
    return history, stats

def wrapper_for_multiprocessing(args):
    """
    這是一個包裝函式，讓 concurrent.futures 可以正確傳遞參數，
    並加入隨機延遲避免 TraCI 通訊埠撞車。
    """
    model_name, model_path, net_file, route_file, steps = args
    time.sleep(np.random.uniform(0.5, 3.0)) 
    history, stats = run_evaluation(model_name, model_path, net_file, route_file, steps)
    return model_name, history, stats

def plot_combined_results(all_histories):
    print("\n📊 正在生成多模型對比綜合圖表 (包含熱力圖)...")
    plt.style.use('ggplot')
    
    # 建立 GridSpec 排版: 3 列 2 行，最後一列的高度稍微矮一點留給熱力圖
    fig = plt.figure(figsize=(16, 15))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8])
    fig.suptitle('Multi-Model Traffic Control Performance & Heatmap', fontsize=18, fontweight='bold')

    # 定義子圖
    ax_arrived = fig.add_subplot(gs[0, 0])
    ax_queue = fig.add_subplot(gs[0, 1])
    ax_wait = fig.add_subplot(gs[1, 0])
    ax_reward = fig.add_subplot(gs[1, 1])
    ax_heatmap = fig.add_subplot(gs[2, :]) # 橫跨底部整排

    model_names = list(all_histories.keys())
    
    # --- 繪製上方 4 張折線圖 ---
    for model_name, history in all_histories.items():
        line_style = '--' if 'Baseline' in model_name else '-'
        line_width = 2.5 if 'Baseline' in model_name else 1.5
        alpha_val = 0.8 if 'Baseline' in model_name else 1.0

        ax_arrived.plot(history['steps'], history['arrived'], label=model_name, linestyle=line_style, linewidth=line_width, alpha=alpha_val)
        ax_queue.plot(history['steps'], history['queue'], label=model_name, linestyle=line_style, linewidth=line_width, alpha=alpha_val)
        ax_wait.plot(history['steps'], history['wait'], label=model_name, linestyle=line_style, linewidth=line_width, alpha=alpha_val)
        
        smoothed_rewards = np.convolve(history['reward'], np.ones(50)/50, mode='valid')
        ax_reward.plot(history['steps'][:len(smoothed_rewards)], smoothed_rewards, label=model_name, linestyle=line_style, linewidth=line_width, alpha=alpha_val)

    ax_arrived.set_title('Cumulative Throughput (Arrived Vehicles)')
    ax_arrived.set_ylabel('Total Vehicles')
    ax_arrived.legend()

    ax_queue.set_title('Total Network Queue Length')
    ax_queue.set_ylabel('Number of Halting Vehicles')
    ax_queue.legend()

    ax_wait.set_title('Total Network Wait Time')
    ax_wait.set_ylabel('Wait Time (Seconds)')
    ax_wait.legend()

    ax_reward.set_title('Agent Reward Over Time (50-Step Moving Average)')
    ax_reward.set_ylabel('Reward')
    ax_reward.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax_reward.legend()

    # --- 繪製下方熱力圖 (Heatmap) ---
    # 1. 找出最短的 step 長度，避免不同陣列長度無法組合
    min_length = min(len(history['queue']) for history in all_histories.values())
    
    # 2. 準備熱力圖的二維陣列數據
    queue_data_2d = np.array([all_histories[name]['queue'][:min_length] for name in model_names])
    
    # 3. 繪製熱力圖 (使用 YlOrRd 色票：黃->橘->紅 代表由順暢到壅塞)
    cax = ax_heatmap.imshow(queue_data_2d, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # 4. 設定熱力圖標籤
    ax_heatmap.set_title('Traffic Congestion Heatmap (Queue Length over Time)')
    ax_heatmap.set_xlabel('Simulation Steps (Seconds)')
    ax_heatmap.set_yticks(np.arange(len(model_names)))
    ax_heatmap.set_yticklabels(model_names)
    
    # 加入顏色條 (Colorbar)
    fig.colorbar(cax, ax=ax_heatmap, label='Queue Length (Vehicles)')

    plt.tight_layout()
    plot_filename = "traffic_multi_model_with_heatmap.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"✅ 圖表已成功儲存至: {plot_filename}")

if __name__ == "__main__":
    NET_FILE = r"D:\school\bsd\test\macao.net.xml"
    ROUTE_FILE = r"D:\school\bsd\log\macao_perfect.rou.xml"
    TEST_STEPS = 3600

    # 🌟 在這裡設定你要跑的所有模型
    MODELS_TO_TEST = {
        "Baseline (Fixed-Time)": None,
        "PPO Model (140k Steps)": r"D:\school\bsd\v5_result\ppo_macao_fast_140000_steps.zip",
        "PPO Model (300k Steps)": r"D:\school\bsd\v5_result\ppo_macao_fast_300000_steps.zip",
    }

    all_histories = {}
    all_stats = {}

    tasks = [(name, path, NET_FILE, ROUTE_FILE, TEST_STEPS) for name, path in MODELS_TO_TEST.items()]

    print("==================================================")
    print(f"⚡ 啟動多進程平行測試 (總共 {len(tasks)} 個任務) ⚡")
    print("==================================================")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(wrapper_for_multiprocessing, tasks)
        
        for model_name, history, stats in results:
            if history is not None:
                all_histories[model_name] = history
                all_stats[model_name] = stats

    print("\n==================================================")
    print("🏆 【 最終效能排行榜 】 🏆")
    print("==================================================")
    sorted_models = sorted(all_stats.items(), key=lambda item: item[1]['Traffic Score'], reverse=True)
    for rank, (name, stats) in enumerate(sorted_models, start=1):
        print(f"第 {rank} 名: {name}")
        print(f"   ➜ 總分: {stats['Traffic Score']} | 疏導量: {stats['Throughput']} | 平均排隊: {stats['Avg Queue']:.2f}")
        print("--------------------------------------------------")

    if all_histories:
        plot_combined_results(all_histories)