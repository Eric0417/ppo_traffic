import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import sumolib
import os
import sys

class MacauSumoEnv(gym.Env):
    """
    澳門全局交通控制環境 (極速輕量快取版)
    """
    def __init__(self, net_file, route_file, use_gui=False, min_green=15, max_steps=3600):
        super(MacauSumoEnv, self).__init__()
        
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.min_green = min_green 
        self.max_steps = max_steps
        self.current_step = 0
        
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("請宣告環境變數 'SUMO_HOME'")

        # 解析路網
        self.net = sumolib.net.readNet(net_file)
        self.tl_ids = [tl.getID() for tl in self.net.getTrafficLights()]
        self.num_tls = len(self.tl_ids)
        
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.num_tls * 2,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([2] * self.num_tls)
        self.tl_durations = {tl: 0 for tl in self.tl_ids}
        
        # 🚀 【核心加速】快取車道與相位資訊，避免每個 Step 都去問 TraCI
        self.tl_lanes = {}
        self.tl_num_phases = {}
        
        # 啟動一次極簡無畫面的 SUMO 來快取資料
        traci.start(["sumo", "-n", self.net_file, "--no-warnings", "true"])
        for tl_id in self.tl_ids:
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            self.tl_lanes[tl_id] = list(set(lanes))
            logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
            self.tl_num_phases[tl_id] = len(logics[0].phases) if logics else 0
        traci.close()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.tl_durations = {tl: 0 for tl in self.tl_ids}
        
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1", 
            "--random",
            "--no-step-log", "true", # 🚀 關閉無用的終端機 log，加速運行
            "--no-warnings", "true"
        ]
        
        try:
            traci.close()
        except:
            pass
        
        traci.start(sumo_cmd)
        return self._get_state(), {}

    def step(self, action):
        self.current_step += 1
        
        for i, tl_id in enumerate(self.tl_ids):
            ai_action = action[i]
            if ai_action == 1 and self.tl_durations[tl_id] >= self.min_green:
                current_phase = traci.trafficlight.getPhase(tl_id)
                # 🚀 直接讀取快取好的相位數
                next_phase = (current_phase + 1) % self.tl_num_phases[tl_id]
                traci.trafficlight.setPhase(tl_id, next_phase)
                self.tl_durations[tl_id] = 0 
            else:
                self.tl_durations[tl_id] += 1

        traci.simulationStep()
        
        state = self._get_state()
        reward, total_queue, total_wait = self._compute_reward()
        arrived_this_step = traci.simulation.getArrivedNumber()
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "total_queue": total_queue,
            "total_wait": total_wait,
            "arrived": arrived_this_step
        }
        
        return state, reward, terminated, truncated, info

    def _get_state(self):
        state = []
        for tl_id in self.tl_ids:
            lanes = self.tl_lanes[tl_id] # 🚀 直接讀取快取，省下大量通訊時間
            halt_cars = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes])
            wait_time = sum([traci.lane.getWaitingTime(lane) for lane in lanes])
            state.extend([halt_cars, wait_time])
        return np.array(state, dtype=np.float32)

    def _compute_reward(self):
        w1 = 1.0
        w2 = 0.1
        total_queue = 0
        total_wait = 0
        
        for tl_id in self.tl_ids:
            lanes = self.tl_lanes[tl_id] # 🚀 直接讀取快取
            total_queue += sum([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes])
            total_wait += sum([traci.lane.getWaitingTime(lane) for lane in lanes])
        
        reward = -(w1 * total_queue + w2 * total_wait)
        return reward, total_queue, total_wait

    def close(self):
        try:
            traci.close()
        except:
            pass