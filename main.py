import pandas as pd
import torch
import traci
import os
import sys
from sumo_env import Sumo_Env
from stable_baselines3 import A2C
from stable_baselines3 import PPO, SAC, TD3, DQN
from heuristic import Heuristic
from data import Collector


from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

torch.manual_seed(4396)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('SUMO_HOME is In Environment!')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


# env = Sumo_Env(sumo_config='./sumo-rl.sumocfg', GUI=True)
env = Sumo_Env(sumo_config='./sumo-rl.sumocfg', GUI=False)
# env_checker.check_env(env)
collector = Collector(env)
collector.collect()


# model = A2C("MlpPolicy", env, verbose=1)
# model = Heuristic()
# obs = env.get_state()
# rewas = []
# for t in range(3600):
#     # actio, _state = model.predict(obs, deterministic=True)
#     actio = model.choose_action(obs)
#     obs, reward, done, _, = env.step(actio)
#     rewas.append(reward)
#     # print(actio,traci.trafficlight.getPhase('0'))
#     t += 1
# avg_rewar = sum(rewas) / len(rewas)
# print('Average Reward: ', avg_rewar)

# new_logger = configure('/log',["stdout", "csv"])
# model = PPO("MlpPolicy", env, verbose=1)
# model.set_logger(new_logger)
# model.learn(total_timesteps=200000)
# model.save('PPO')
