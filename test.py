import os
import sys
from sumo_env import Sumo_Env
from heuristic import Heuristic
import traci
import random
import numpy as np
import pandas as pd
from stable_baselines3 import DQN,A2C,PPO
import matplotlib.pyplot as plt
import torch

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('SUMO_HOME is In Environment!')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


env = Sumo_Env(sumo_config=r"F:\py39_project\sumo_project\signal_project\sumo-rl.sumocfg", GUI=True)



'''
model = PPO.load(env=env,path=r"F:\py39_project\sumo_project\signal_project\ppo1.zip")
env.launch_env()
#obs = env.get_state()
obs = env.reset()

i = 0
queue_length_list_l = []
queue_length_list_r = []
cumulative_queue_length_l = 0
cumulative_queue_length_r = 0


while i < 3600:
    if i < 1000 :
        traci.simulationStep()
        queue_length_left, queue_length_right = env.get_queue_length()
        cumulative_queue_length_l += queue_length_left
        cumulative_queue_length_r += queue_length_right

    else:
        obs = env.get_state()
        action, obs = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        queue_length_left, queue_length_right = env.get_queue_length()
        cumulative_queue_length_l += queue_length_left
        cumulative_queue_length_r += queue_length_right

    queue_length_list_l.append(cumulative_queue_length_l)
    queue_length_list_r.append(cumulative_queue_length_r)
    i += 1

traci.close()
'''

# queue_length_list_l= pd.DataFrame(queue_length_list_l)
# queue_length_list_r= pd.DataFrame(queue_length_list_r)
#
# queue_length_list_l.to_csv(r"F:\py39_project\sumo_project\signal_project\data_RL_control\PPO_l1_3600.csv",header=True)
# queue_length_list_r.to_csv(r"F:\py39_project\sumo_project\signal_project\data_RL_control\PPO_r1_3600.csv",header=True)

# 不控制
env.launch_env()
i = 0
queue_length_list_l = []
queue_length_list_r = []
cumulative_queue_length_l = 0
cumulative_queue_length_r = 0

while i < 3600:
    traci.simulationStep()
    queue_length_left, queue_length_right = env.get_queue_length()
    cumulative_queue_length_l += queue_length_left
    cumulative_queue_length_r += queue_length_right
    queue_length_list_l.append(cumulative_queue_length_l)
    queue_length_list_r.append(cumulative_queue_length_r)
    i += 1

traci.close()
# queue_length_list_l= pd.DataFrame(queue_length_list_l)
# queue_length_list_r= pd.DataFrame(queue_length_list_r)
#
# queue_length_list_l.to_csv(r"F:\py39_project\sumo_project\signal_project\data_no_control\no_control_l_3600.csv",header=True)
# queue_length_list_r.to_csv(r"F:\py39_project\sumo_project\signal_project\data_no_control\no_control_r_3600.csv",header=True)


# 静态图
'''
l = range(1,len(queue_length_list_l)+1)
queue_length_list_l =np.array(queue_length_list_l)
queue_length_list_r =np.array(queue_length_list_r)
plt.rcParams['font.sans-serif']=['SimHei']
# plt.xticks([0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
# plt.yticks([0,20000,40000,60000,80000,100000,120000,140000])
plt.xlabel('仿真时间')
plt.ylabel('累积排队数')
plt.grid(True)
plt.plot(l,queue_length_list_l, color='b', lw=1, label="左边车道")
plt.plot(l,queue_length_list_r, color='r', lw=1, label="右边车道")

plt.legend(loc='best')
plt.show()
'''