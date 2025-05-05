import copy
import math
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans

import torch
import torch.nn as nn

from ..strategy import Strategy
import os

class KSAS(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)

        g_net = self.net
        l_net = deepcopy(self.net)
        load_path = os.path.join('temp', 'user'+str(user_idx)+'.pt')
        l_net.load_state_dict(torch.load(load_path), strict=False)
        #l_net = self.training_local_only(label_idxs)

        label_count = self.get_label_count(label_idxs)
        U = self.get_discrepancy(l_net, g_net, unlabel_idxs, label_count)

        return unlabel_idxs[U.sort()[1][-n_query:]]