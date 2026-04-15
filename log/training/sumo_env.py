import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
from sumolib import checkBinary
import os

class SumoEnv(gym.Env):
    def __init__(self):
        super(SumoEnv, self).__init__()
        
        self.sumo_cmd = [checkBinary('sumo'), "-c", r"D:\school\bsd\log\macao.sumocfg", "--lateral-resolution", "0.8"]
        self.step_count = 0
        self.max_steps = 3600  
        
        self.min_green_time = 10 
        self.max_green_time = 60 
        
        print("🔍 正在啟動 SUMO 以掃描全澳門路口資訊...")
        traci.start(self.sumo_cmd)
        
        raw_tl_ids = sorted(traci.trafficlight.getIDList())
        self.tl_ids = []
        self.lanes_dict = {}
        self.num_phases_dict = {}
        self.timers = {}
        self.total_lanes = 0
        
        # 🌟 修正：加入過濾機制，剔除無法切換的「假紅綠燈」
        for tl in raw_tl_ids:
            logics = traci.trafficlight.getAllProgramLogics(tl)
            num_phases = len(logics[0].phases) if logics else 0
            
            # 只有階段數 >= 2 的路口，才有讓 AI 控制的意義
            if num_phases >= 2:
                self.tl_ids.append(tl)
                self.num_phases_dict[tl] = num_phases
                
                lanes = sorted(list(set(traci.trafficlight.getControlledLanes(tl))))
                self.lanes_dict[tl] = lanes
                self.total_lanes += len(lanes)
                self.timers[tl] = 0 
            
        self.num_agents = len(self.tl_ids)
        traci.close()
        
        print(f"✅ 掃描與過濾完成！剔除無效路口後，共有 {self.num_agents} 個有效受控路口，總計 {self.total_lanes} 條受控車道。")

        # 動作空間：長度為 num_agents 的陣列
        self.action_space = spaces.MultiDiscrete([2] * self.num_agents)
        
        # 感官空間：[所有車道擁擠度] + [所有路口當前燈號]
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
        
        try:
            traci.close()
        except:
            pass
            
        traci.start(self.sumo_cmd)
        
        initial_state = np.zeros(self.total_obs_length, dtype=np.float32)
        
        idx = self.total_lanes
        for i, tl in enumerate(self.tl_ids):
            initial_state[idx + i] = traci.trafficlight.getPhase(tl)
            
        return initial_state, {}

    def step(self, actions):
        action_penalty_total = 0
        
        for i, tl in enumerate(self.tl_ids):
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
        
        raw_penalty = total_halting + (0.5 * total_waiting_time)
        scaled_penalty = raw_penalty / 100.0 
        
        reward = -scaled_penalty + action_penalty_total
        
        terminated = False
        truncated = bool(self.step_count >= self.max_steps)
        
        return next_state, reward, terminated, truncated, {}

    def close(self):
        try:
            traci.close()
        except:
            pass