import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from sumo_env import MacauSumoEnv

def train_new_model():
    print("==================================================")
    print("🚀 啟動澳門交通 AI (輕量快攻版 - 搶時間專用)")
    print("==================================================")

    # 1. 設定檔案路徑
    NET_FILE = r"D:\school\bsd\test\macao.net.xml"
    ROUTE_FILE = r"D:\school\bsd\log\macao_perfect.rou.xml"
    MODEL_DIR = r"D:\school\bsd\v5_result"
    LOG_DIR = r"D:\school\bsd\logs_v5"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 2. 初始化全新環境
    env = MacauSumoEnv(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False,  
        min_green=15,
        max_steps=3600  
    )
    env = Monitor(env)

    # 3. 建立【輕量化】PPO 模型
    # 🚀 降級網路：[256, 256] 已經足夠處理這種控制問題，運算速度快很多
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1, 
        tensorboard_log=LOG_DIR,
        
        # 🚀 【快訓參數微調】
        learning_rate=0.0003,  # 標準 PPO 學習率，學得比較快
        n_steps=2048,          # 縮短到 2048 步就更新一次模型，不用等太久
        batch_size=256,        # 配合 n_steps 縮小
        ent_coef=0.01,         # 降低一點探索率，盡快收斂到可用解
        gamma=0.99,            
        device='cpu'          
    )

    # 4. 設定存檔點 (每 2 萬步存一次，因為總步數減少了)
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path=MODEL_DIR,
        name_prefix="ppo_macao_fast"
    )

    # 5. 實務訓練目標：先跑 50 萬步 (500k) 即可見效
    TOTAL_STEPS = 500000
    print(f"[-] 模型已建立，準備進行 {TOTAL_STEPS} 步的高速訓練...")
    
    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_callback)
        print("\n[+] 訓練圓滿結束！")
        
        final_model_path = os.path.join(MODEL_DIR, "ppo_macao_fast_final.zip")
        model.save(final_model_path)
        print(f"[+] 最終模型已儲存至: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n[!] 訓練被手動中斷。")
        interrupt_model_path = os.path.join(MODEL_DIR, "ppo_macao_fast_interrupted.zip")
        model.save(interrupt_model_path)
        print(f"[+] 已儲存中斷前的模型至: {interrupt_model_path}")
    finally:
        env.close()

if __name__ == "__main__":
    train_new_model()