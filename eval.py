from d3rlpy.algos import DiscreteCQL,DiscreteBCQ,DiscreteBC,DiscreteSAC
import d3rlpy
import os
import sys
import torch
from sumo_env import Sumo_Env
import matplotlib.pyplot as plt
import traci
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from d3rlpy.wrappers.sb3 import SB3Wrapper


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('SUMO_HOME is In Environment!')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

env = Sumo_Env(sumo_config='./sumo-rl.sumocfg', GUI=True)
# model = DiscreteSAC()
# model = DiscreteCQL()
model = DiscreteBCQ()
model.build_with_env(env)
model.load_model(r'F:\py39_project\sumo_project\offline\hig_ve_model\bcq.pt')

env.launch_env()
#obs = env.get_state()

i = 0
wait_time_list_l = []
wait_time_list_r = []
cumulative_wait_time_l = 0
cumulative_wait_time_r = 0

while i < 7000:
    if i < 1000:
        traci.simulationStep()
        wait_time_l, wait_time_r = env.get_wait_time()
        cumulative_wait_time_l += wait_time_l
        cumulative_wait_time_r += wait_time_r

    else:
        obs = env.get_state()

        # obs = np.expand_dims(obs,axis=0)
        obs = torch.tensor(obs)
        obs = torch.unsqueeze(obs,0)
        action = model.predict(obs)
        # phase = traci.trafficlight.getPhase('0')
        # print(phase,action)
        obs, reward, done, _ = env.step(action)
        wait_time_l, wait_time_r = env.get_wait_time()
        cumulative_wait_time_l += wait_time_l
        cumulative_wait_time_r += wait_time_r

    wait_time_list_l.append(cumulative_wait_time_l)
    wait_time_list_r.append(cumulative_wait_time_r)
    i += 1

traci.close()

'''

i = 0
wait_time_list_l = []
wait_time_list_r = []
cumulative_wait_time_l = 0
cumulative_wait_time_r = 0

while i < 3600:

    traci.simulationStep()
    wait_time_l, wait_time_r = env.get_wait_time()
    cumulative_wait_time_l += wait_time_l
    cumulative_wait_time_r += wait_time_r
    wait_time_list_l.append(cumulative_wait_time_l)
    wait_time_list_r.append(cumulative_wait_time_r)

    i += 1

traci.close()
'''

l = range(1,len(wait_time_list_l)+1)
wait_time_list_l =np.array(wait_time_list_l)
wait_time_list_r =np.array(wait_time_list_r)
plt.rcParams['font.sans-serif']=['SimHei']
plt.xticks([0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
# plt.yticks([0,20000,40000,60000,80000,100000,120000,140000])
plt.xlabel('仿真时间')
plt.ylabel('车辆累计等待时间')
plt.grid(True)
plt.plot(l,wait_time_list_l, color='b', lw=1, label="左边车道")
plt.plot(l,wait_time_list_r, color='r', lw=1, label="右边车道")

plt.legend(loc='best')
plt.show()
