# import gymnasium as gym
# from gymnasium import spaces
import gym
import numpy as np
import traci
from arena import integer_to_one_hot
import sys


class Sumo_Env(gym.Env):
    """Custom Sumo Environment that follows gym interface."""

    def __init__(self, sumo_config, sim_durat=3600, GUI=False):
        super().__init__()
        # Define action and observation space, they must be gym.spaces objects
        # self.action_space = spaces.MultiDiscrete()
        self.action_space = gym.spaces.Discrete(2)
        # self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32)
        self.observation_space = gym.spaces.Box(
            low=0, high=80, shape=(4, ), dtype=np.float64)

        self.sumo_config = sumo_config  # sumo config 文件
        self.sim_durat = sim_durat
        self.GUI = GUI
        # the minimum green time for each phase
        self.min_green = 10

        self.vehicles = dict()
        self.last_measure = 0.0
        # self.lane_list = ['11.30_0', '11.30_1', '11_0', '11_1', '11_2', '6_0', '6_1', '6_2', '15_0', '15_1', '15_2', 'E1_0', 'E0_2', '7_0', '8_0', '9_0', '5f_0']
        # self.lane_list_left = ['11.30_0', '11.30_1', '11_0', '11_1', '11_2', '6_0','6_1', '6_2']
        # self.lane_list_right = ['7_0', '8_0', '9_0']
        self.lane_list_left_first = ['11.30_0', '11.30_1','11_0','11_1','11_2']
        self.lane_list_left_second = ['6_0','6_1','6_2']
        self.lane_list_left_total = ['11.30_0', '11.30_1','11_0','11_1','11_2','6_0','6_1','6_2']
        self.lane_list_right_first = ['8_0', '9_0']
        self.lane_list_right_second = ['7.69_0']
        self.lane_list_right_total = ['7_0' , '8_0', '9_0']

        # tracking last phase

        # track phase duration
        self.phase_duration_l = 0
        self.phase_duration_r = 0

        self.last_phase_idx = 0
        self.yellow_time_l = 0
        self.yellow_time_r = 0

    def step(self, actio):
        # self.phase_duration += 1

        if actio is not None:
            self.set_phase(actio)

        traci.simulationStep()
        obser = self.get_state()
        obser = np.array(obser)
        # get reward
        rewar = self.get_reward()
        # self.update()
        return obser, rewar, False, {}

    def reset(self, seed=None, options=None):
        self.phase_duration_l = 0
        self.phase_duration_r = 0
        self.close()
        self.launch_env()
        obser = self.get_state()
        return np.array(obser)

    def close(self):
        traci.close()
        self.launch_env_flag = False
        sys.stdout.flush()

   # sumo-related for RL
    def set_phase(self, actio):
        # plan = 'GGr' if actio == 0 else 'rrG'
        a = 0 if actio == 0 else 2
        curr_phase_id = traci.trafficlight.getPhase('0')

        if curr_phase_id == 0:
            self.phase_duration_l += 1
        elif curr_phase_id == 2:
            self.phase_duration_r += 1

        if (self.phase_duration_r >= self.min_green and a != curr_phase_id and a == 0) or (curr_phase_id == 3):

            if self.yellow_time_r < 3:
                traci.trafficlight.setPhase('0', 3)
                self.yellow_time_r += 1
            else:
                traci.trafficlight.setPhase('0', 0)
                self.phase_duration_r = 0
                self.yellow_time_r = 0

        elif (self.phase_duration_l >= self.min_green and a != curr_phase_id and a == 2) or (curr_phase_id == 1):

            if self.yellow_time_l < 3:
                traci.trafficlight.setPhase('0', 1)
                self.yellow_time_l += 1
            else:
                traci.trafficlight.setPhase('0', 2)
                self.phase_duration_l = 0
                self.yellow_time_l = 0

        else:
            traci.trafficlight.setPhase('0', curr_phase_id)
        # traci.trafficlight.setPhase('0', idx)


    def get_phase_index(self):
        curr_phase_idx = traci.trafficlight.getPhase('0')
        phase_one_hot = integer_to_one_hot(curr_phase_idx)
        return phase_one_hot

    def get_state(self):
        # 1. phase_one_hot: one-hot encoded vector indicating the current active (green?) phase
        # curr_phase_idx = traci.trafficlight.getPhase('0')
        # phase_one_hot = integer_to_one_hot(curr_phase_idx)
        # 2. min_green: a binary variable indicating whether min_green seconds have already passed in the current phase
        # min_green = 1 if self.phase_duration >= self.min_green else 0
        # 3.  lane density
        first_left_veh_num, second_left_veh_num, first_right_veh_num, second_right_veh_num = self.get_lane_veh_num()

        first_left_veh_num /= 10
        second_left_veh_num /= 10
        first_right_veh_num /= 10
        second_right_veh_num /= 10
        # combine all the features
        state = [first_left_veh_num] + [second_left_veh_num] + [first_right_veh_num] + [second_right_veh_num]

        return state

    def get_reward(self):

        total_vehicle_left_first,total_vehicle_left_second,total_vehicle_right_first,total_vehicle_right_second = self.get_lane_veh_num()
        total_vehicle_left = total_vehicle_left_first + total_vehicle_left_second
        total_vehicle_right = total_vehicle_right_first + total_vehicle_right_second

        # if total_vehicle_left >= 100:
        #     rewar = -2000
        #
        # else:
        rewar = -(total_vehicle_left + total_vehicle_right * 2)

        return rewar

    def get_lane_veh_num(self):

        total_vehicle_left_first = 0
        total_vehicle_left_second = 0
        total_vehicle_right_first = 0
        total_vehicle_right_second = 0

        for id in self.lane_list_left_first:
            total_vehicle_left_first += traci.lane.getLastStepVehicleNumber(id)

        for id in self.lane_list_left_second:
            total_vehicle_left_second += traci.lane.getLastStepVehicleNumber(id)

        for id in self.lane_list_right_first:
            total_vehicle_right_first += traci.lane.getLastStepVehicleNumber(id)

        for id in self.lane_list_right_second:
            total_vehicle_right_second += traci.lane.getLastStepVehicleNumber(id)

        return total_vehicle_left_first,total_vehicle_left_second,total_vehicle_right_first,total_vehicle_right_second

    # def get_queue_length(self):
    #     queue_left = 0
    #     queue_right = 0
    #     for lane in self.lane_list_left:
    #         queue_left += traci.lane.getLastStepHaltingNumber(lane)
    #
    #     for lane in self.lane_list_right:
    #         queue_right += traci.lane.getLastStepHaltingNumber(lane)
    #
    #     return queue_left, queue_right


    def get_wait_time(self):
        wait_time_l = 0
        wait_time_r = 0

        for lane in self.lane_list_left_total:
            veh_list_l = traci.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list_l:
                acc_l = traci.vehicle.getWaitingTime(veh)
                if acc_l != 0:
                    wait_time_l += 1
                else:
                    wait_time_l += 0

        for lane in self.lane_list_right_total:
            veh_list_r = traci.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list_r:
                acc_r = traci.vehicle.getWaitingTime(veh)
                if acc_r != 0:
                    wait_time_r += 1
                else:
                    wait_time_r += 0

        return wait_time_l,wait_time_r

    def launch_env(self):
        if self.GUI:
            sumo_gui = 'sumo-gui'
        else:
            sumo_gui = 'sumo'
        traci.start([
            sumo_gui,
            "-c", self.sumo_config,
            "--no-warnings",
            "--seed", "5"])
        self.launch_env_flag = True
