from d3rlpy.algos import DiscreteBC,DiscreteBCQ,DiscreteCQL,DQN,DoubleDQN,NFQ,DiscreteSAC,DiscreteRandomPolicy
from d3rlpy.wrappers.sb3 import to_mdp_dataset
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics.scorer import td_error_scorer
import torch
torch.manual_seed(4396)

dataset = MDPDataset.load(r'F:\py39_project\sumo_project\offline\hig_ve\MDP_data_10000_30_1.h5')

# cql = DiscreteCQL(use_gpu=True,learning_rate=0.00003)
# cql.fit(
#     dataset,
#     n_epochs=15,
#     scorers={
#         'td_error': td_error_scorer,
#     },
# )
# cql.save_model('cql.pt')
bcq= DiscreteBCQ(learning_rate=0.00003)  #learning_rate=0.00006
bcq.fit(
    dataset,
    n_steps_per_epoch=3000000,
    n_epochs=2,
    scorers={
        'td_error': td_error_scorer,
    },
)

bcq.save_model(r'F:\py39_project\sumo_project\offline\hig_ve_model\bcq.pt')