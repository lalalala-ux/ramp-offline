import d3rlpy
import numpy as np
from d3rlpy.datasets import MDPDataset
from d3rlpy.algos import DiscreteCQL
from d3rlpy.metrics.scorer import td_error_scorer
from sklearn.model_selection import train_test_split
from copy import copy


class Collector:
    def __init__(self, env):
        self.env = env

    def collect(self):
        self.env.launch_env()
        # env_checker.check_env(env)
        obser_list, actio_list, rewar_list, next_obs_list, termi_list = [], [], [], [], []
        for episode in range(2):
            self.env.reset()
            next_obser = self.env.get_state()
            for t in range(3600):
                # actio = model.choose_action(obs)
                obser = copy(next_obser)
                obser_list.append(obser)
                next_obser, rewar, done, _, = self.env.step(actio=None)
                next_phase_index = next_obser[0:4].argmax()
                # 0-GGr, 2-rrG
                # 1-yyr, 3-rry
                if next_phase_index in [1, 2]:
                    actio = 1
                elif next_phase_index in [0, 3]:
                    actio = 0
                actio_list.append(actio)
                rewar_list.append(rewar)
                if t == 3599:
                    termi_list.append(1)
                else:
                    termi_list.append(0)

        obses = np.array(obser_list).reshape(-1, 6)
        actis = np.array(actio_list).reshape(-1, 1)
        rewas = np.array(rewar_list)
        terms = np.array(termi_list)

        dataset = MDPDataset(observations=obses, actions=actis,
                             rewards=rewas, terminals=terms, discrete_action=True)
        print('episode num:', len(dataset))

        cql = DiscreteCQL()
        cql.fit(
            dataset,
            n_epochs=5,
            scorers={
                'td_error': td_error_scorer,
            },
        )
