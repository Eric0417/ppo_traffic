import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
from sumolib import checkBinary
import os

class MacauSumoEnv(gym.Env):
    # 接收 score_model.py 傳進來的參數，保持介面相容
    def __init__(self, net_file=None, route_file=None, use_gui=False, min_green=10, max_steps=3600, scale=1.2):
        super(MacauSumoEnv, self).__init__()
        
        self.use_gui = use_gui
        self.step_count = 0
        self.max_steps = max_steps  
        self.scale = scale
        
        self.min_green_time = min_green 
        self.max_green_time = 60 
        
        # 根據 use_gui 決定要啟動有畫面還是無畫面的 SUMO
        sumo_binary = checkBinary('sumo-gui') if self.use_gui else checkBinary('sumo')
        
        # ★ 這裡維持你訓練時的配置，但加入了測試必須的參數
        self.sumo_cmd_base = [
            sumo_binary, 
            "-c", r"D:\school\bsd\log\macao.sumocfg", # 使用你指定的 cfg
            "--lateral-resolution", "0.8",
            "--time-to-teleport", "300",              # 防死鎖機制
            "--scale", str(self.scale),               # 降壓測試倍率
            "--waiting-time-memory", "10000"
        ]
        
        print("🔍 正在啟動 SUMO 以掃描全澳門路口資訊 (測試模式)...")
        traci.start(self.sumo_cmd_base)
        
        raw_tl_ids = sorted(traci.trafficlight.getIDList())
        self.tl_ids = []
        self.lanes_dict = {}
        self.num_phases_dict = {}
        self.timers = {}
        self.total_lanes = 0
        
        # 🌟 保持與訓練完全相同的過濾機制
        for tl in raw_tl_ids:
            logics = traci.trafficlight.getAllProgramLogics(tl)
            num_phases = len(logics[0].phases) if logics else 0
            
            if num_phases >= 2:
                self.tl_ids.append(tl)
                self.num_phases_dict[tl] = num_phases
                
                lanes = sorted(list(set(traci.trafficlight.getControlledLanes(tl))))
                self.lanes_dict[tl] = lanes
                self.total_lanes += len(lanes)
                self.timers[tl] = 0 
            
        self.num_agents = len(self.tl_ids)
        traci.close()
        
        print(f"✅ 掃描與過濾完成！有效路口: {self.num_agents} 個，受控車道: {self.total_lanes} 條。")

        # 動作空間：與訓練環境完全一致
        self.action_space = spaces.MultiDiscrete([2] * self.num_agents)
        
        # 觀測空間：與訓練環境完全一致
        self.total_obs_length = self.total_lanes + self.num_agents
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1000.0, 
            shape=(self.total_obs_length,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        for tl in self.tl_ids:
            self.timers[tl] = 0
            
        current_seed = seed if seed is not None else 42
        
        # 每次重置時加入隨機種子，確保 Baseline 和 AI 測試的一致性
        cmd = self.sumo_cmd_base + ["--seed", str(current_seed)]
        
        try:
            traci.close()
        except:
            pass
            
        traci.start(cmd)
        
        initial_state = np.zeros(self.total_obs_length, dtype=np.float32)
        
        idx = self.total_lanes
        for i, tl in enumerate(self.tl_ids):
            initial_state[idx + i] = traci.trafficlight.getPhase(tl)
            
        return initial_state, {}

    def step(self, actions):
        action_penalty_total = 0
        
        # ★ 保持與訓練完全相同的紅綠燈控制邏輯
        for i, tl in enumerate(self.tl_ids):
            # 如果測試 Baseline 傳入 None，就什麼動作都不做 (純觀察)
            if actions is None:
                break
                
            action = actions[i]
            current_phase = traci.trafficlight.getPhase(tl)
            phase_state = traci.trafficlight.getRedYellowGreenState(tl)
            
            is_yellow = 'y' in phase_state.lower()
            
            if is_yellow:
                self.timers[tl] = 0 
            else:
                if action == 1 and self.timers[tl] >= self.min_green_time:
                    next_phase = (current_phase + 1) % self.num_phases_dict[tl]
                    traci.trafficlight.setPhase(tl, next_phase)
                    self.timers[tl] = 0
                    action_penalty_total -= 0.1
                elif self.timers[tl] >= self.max_green_time:
                    next_phase = (current_phase + 1) % self.num_phases_dict[tl]
                    traci.trafficlight.setPhase(tl, next_phase)
                    self.timers[tl] = 0
                    action_penalty_total -= 0.1
                else:
                    self.timers[tl] += 1
                    
        traci.simulationStep()
        self.step_count += 1
        
        lane_halts = []
        total_halting = 0
        total_waiting_time = 0
        
        # ★ 保持與訓練完全相同的狀態特徵提取
        for tl in self.tl_ids:
            for lane in self.lanes_dict[tl]:
                halt_num = traci.lane.getLastStepHaltingNumber(lane)
                lane_halts.append(halt_num)
                total_halting += halt_num
                total_waiting_time += traci.lane.getWaitingTime(lane)

        new_phases = []
        for tl in self.tl_ids:
            new_phases.append(traci.trafficlight.getPhase(tl))
            
        state_list = lane_halts + new_phases
        next_state = np.array(state_list, dtype=np.float32)
        
        # ★ 保持與訓練完全相同的獎勵計算
        raw_penalty = total_halting + (0.5 * total_waiting_time)
        scaled_penalty = raw_penalty / 100.0 
        reward = -scaled_penalty + action_penalty_total
        
        terminated = False
        truncated = bool(self.step_count >= self.max_steps)
        
        # ★ 關鍵新增：抓取測試計分板所需的數據
        arrived_this_step = traci.simulation.getArrivedNumber()
        info = {
            "total_queue": total_halting,
            "total_wait": total_waiting_time,
            "arrived": arrived_this_step
        }
        
        return next_state, reward, terminated, truncated, info

    def close(self):
        try:
            traci.close()
        except:
            pass