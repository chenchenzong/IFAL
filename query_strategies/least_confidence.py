import copy
import numpy as np
from copy import deepcopy
from .strategy import Strategy
import os

import torch
import torch.nn as nn


class LeastConfidence(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)
        
        if self.args.query_model_mode == "global":
            probs = self.predict_prob(unlabel_idxs, self.net)
        elif self.args.query_model_mode == "local_only":
            local_net = deepcopy(self.net)
            load_path = os.path.join('temp', 'user'+str(user_idx)+'.pt')
            local_net.load_state_dict(torch.load(load_path), strict=False)
            #local_net = self.training_local_only(label_idxs)
            probs = self.predict_prob(unlabel_idxs, local_net)
        
        U = probs.max(1)[0]
        return unlabel_idxs[U.sort()[1][:n_query]]