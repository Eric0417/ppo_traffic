from stable_baselines3 import PPO
from sumo_env import SumoEnv
import time

# 1. 開啟道館 (這次會有 GUI 畫面彈出來)
env = SumoEnv()

print("🧠 正在載入剛結業的 AI 交通警察模型...")
# 2. 讀取我們剛剛存檔的 AI 大腦 (.zip)
model = PPO.load(r"D:\school\bsd\ppo_macao_v2_final.zip")

# 3. 重置環境，獲取初始的塞車狀況 (0 輛車)
obs, info = env.reset()
print("🚦 AI 已上線，開始接管高士德路口！請觀看 SUMO 畫面！")

# 4. 讓 AI 連續指揮 3600*24 步 (24 小時)
for i in range(3600*24):
    # deterministic=True 代表讓 AI 發揮它「確定的」最佳實力，不再做隨機探索
    action, _states = model.predict(obs, deterministic=True)
    
    # 將 AI 決定的動作傳給模擬器執行，並獲取新的塞車狀況 (obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 如果回合結束 (滿 3600*24 步)，就跳出迴圈
    if terminated or truncated:
        print("🏁 模擬測試結束！")
        break

# 關閉環境
env.close()