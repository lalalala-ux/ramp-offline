import d3rlpy
import numpy as np
import torch
from d3rlpy.datasets import MDPDataset
from d3rlpy.algos import DiscreteCQL,DiscreteBCQ,DiscreteBC

from d3rlpy.metrics.scorer import td_error_scorer
from sklearn.model_selection import train_test_split
from copy import copy

# 这里只保存数据
class Collector:
    def __init__(self, env):

        self.env = env
        # self.env.launch_env()

    def collect(self):
        # env_checker.check_env(env)
        obser_list, actio_list, rewar_list, next_obs_list, termi_list = [], [], [], [], []
        for episode in range(30):
            self.env.reset()
            next_obser = self.env.get_state()

            for t in range(10000):
                # actio = model.choose_action(obs)
                obser = copy(next_obser)
                obser_list.append(obser)
                next_obser, rewar, done, _, = self.env.step(actio=None)
                phase_indx = self.env.get_phase_index()
                phase_indx = torch.tensor(phase_indx)
                next_phase_index = phase_indx.argmax()
                # 0-GGr, 2-rrG
                # 1-yyr, 3-rry
                # if next_phase_index in [2,3]:
                #     actio = 1
                # elif next_phase_index in [0,1]:
                #     actio = 0
                if next_phase_index == 2:
                    actio = 1
                elif next_phase_index == 0:
                    actio = 0

                actio_list.append(actio)
                rewar_list.append(rewar)
                if t == 9999:
                    termi_list.append(1)
                else:
                    termi_list.append(0)

        obses = np.array(obser_list).reshape(-1, 4)
        actis = np.array(actio_list).reshape(-1, 1)
        rewas = np.array(rewar_list)
        terms = np.array(termi_list)

        dataset = MDPDataset(observations=obses, actions=actis,
                             rewards=rewas, terminals=terms, discrete_action=True)

        dataset.dump(r'F:\py39_project\sumo_project\offline\hig_ve\MDP_data_10000_30_1.h5')

        # print('episode num:', len(dataset))

        # cql = DiscreteCQL(use_gpu=True,learning_rate=0.00003)
        # bcq = DiscreteBCQ(learning_rate=0.0003)
        # bcq.fit(
        #     dataset,
        #
        #     n_epochs=10,
        #     scorers={
        #         'td_error': td_error_scorer,
        #     },
        # )
        #
        # bcq.save_model('bcq.pt')
