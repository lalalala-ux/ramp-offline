import numpy as np
from collections import defaultdict, namedtuple
import torch
import random


class Heuristic:
    def __init__(self) -> None:
        super(Heuristic, self).__init__()

    def choose_action(self, state):
        if state[5] >= state[6]:
            actio = 0
        else:
            actio = 2
        return actio
