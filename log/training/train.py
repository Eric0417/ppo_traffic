import os
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from sumo_env import SumoEnv

# ==========================================
# 1. 建立資料夾
# ==========================================
os.makedirs("./models", exist_ok=True)
# 開啟全新的 log 資料夾，確保從零開始的圖表乾淨
os.makedirs("./logs_v3", exist_ok=True) 

# ==========================================
# 2. 開啟道館
# ==========================================
env = SumoEnv()

# ==========================================
# 3. 設定自動存檔 (Checkpoint)
# ==========================================
checkpoint_callback = CheckpointCallback(
    save_freq=25000,                # 每 25,000 步存檔一次
    save_path="./models/",
    name_prefix="ppo_macao_v3"      # 存檔的檔名前綴
)

# ==========================================
# 4. 定義全新的 AI 大腦與學習策略 (城市級升級)
# ==========================================
# 動態學習率：隨著訓練進度，學習率會慢慢降到 0，讓 AI 後期學習更穩定
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# 🌟 升級：加大 AI 腦容量以容納全澳門地圖 (512x512x512)
custom_policy_kwargs = dict(
    net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]) 
)

print("🧠 正在初始化 城市級 AI 大腦 (512x512x512 網路 + 動態學習率)...")

# 建立全新的 PPO 模型
model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=linear_schedule(0.0003),
    policy_kwargs=custom_policy_kwargs,
    n_steps=3600,            # 🌟 剛好收集完 1 小時 (3600步) 的資料才進行反思更新
    batch_size=900,          # 🌟 加大 Batch Size，讓全城數據的更新更穩定 (3600能被900整除)
    ent_coef=0.01,           # 保持 1% 的探索好奇心
    verbose=1, 
    device="cuda",           # 🌟 啟動 RTX 3070 Ti 顯卡加速
    tensorboard_log="./logs_v3/" 
)

# ==========================================
# 5. 開始全新訓練
# ==========================================
total_steps = 2000000
print(f"🔥 V3 城市級全網特訓開始！預計進行 {total_steps} 步...")

model.learn(
    total_timesteps=total_steps, 
    callback=checkpoint_callback,   
    tb_log_name="PPO_City_Scale"   
)

# ==========================================
# 6. 儲存最終模型與關閉環境
# ==========================================
print("🎉 200 萬步城市級訓練結束！")
final_model_name = "./models/ppo_macao_v3_512x512_final"
model.save(final_model_name)
print(f"💾 最終模型已成功儲存為 {final_model_name}.zip")

env.close()
print("✅ 環境已安全關閉。")