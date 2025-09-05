import copy
import math
import numpy as np
from copy import deepcopy

import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import functools

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from scipy.stats import wasserstein_distance
from skimage.filters import threshold_otsu

from ..strategy import Strategy

def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)


def symmetric_kl_div(p, q):
    return kl_div(p, q) + kl_div(q, p)


def js_div(p, q):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

class IFAL(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)

        g_net = self.net

        l_net = deepcopy(self.net)
        load_path = os.path.join('temp', 'user'+str(user_idx)+'.pt')
        l_net.load_state_dict(torch.load(load_path), strict=False)

        l_net_kd = self.training_local_only_KD(label_idxs)

        g_labelArr_all = self.get_rknn_label(label_idxs, unlabel_idxs)

        g_features_l = self.get_embedding(label_idxs, GPU=True)
        g_features_l = F.normalize(g_features_l, dim=1)
        g_features_un, g_c_probs = self.get_embedding_and_prob(unlabel_idxs, GPU=True)
        g_features_un = F.normalize(g_features_un, dim=1)
        dists_all = torch.mm(torch.vstack((g_features_l,g_features_un)), g_features_un.t())
        dists_all[torch.arange(g_features_l.size()[0], dists_all.size()[0]), torch.arange(g_features_un.size()[0])] = -1
        _, top_k_index = dists_all.topk(250, dim=1, largest=True, sorted=True) ## Top-K similar scores and corresponding indexes
        rknn_logits = torch.ones(g_features_un.shape[0], self.args.num_classes, dtype=torch.long).to(self.args.device)
        for i in range(self.args.num_classes):
            unique_indices, counts = torch.unique(top_k_index[np.asarray(g_labelArr_all)==i], return_counts=True)
            rknn_logits[unique_indices,i] += counts
        rknn_logits = rknn_logits.cpu()
        g_probs1 = rknn_logits/(rknn_logits.sum(1).reshape(-1,1))
        g_probs2 = rknn_logits/(rknn_logits.sum(1).reshape(-1,1))
        g_pred1 = g_probs1.max(1)[1]
        g_pred2 = g_probs2.max(1)[1]
        g_c_pred = g_c_probs.max(1)[1].cpu()

        l_labelArr_all = self.get_rknn_label(label_idxs, unlabel_idxs, l_net)
        
        l_features_l = self.get_embedding(label_idxs, l_net, GPU=True)
        l_features_l = F.normalize(l_features_l, dim=1)
        l_features_un, l_c_probs = self.get_embedding_and_prob(unlabel_idxs, l_net, GPU=True)
        l_features_un = F.normalize(l_features_un, dim=1)
        dists_all = torch.mm(torch.vstack((l_features_l,l_features_un)), l_features_un.t())
        dists_all[torch.arange(l_features_l.size()[0], dists_all.size()[0]), torch.arange(l_features_un.size()[0])] = -1
        _, top_k_index = dists_all.topk(250, dim=1, largest=True, sorted=True) ## Top-K similar scores and corresponding indexes
        rknn_logits = torch.ones(l_features_un.shape[0], self.args.num_classes, dtype=torch.long).to(self.args.device)
        l_rknn_logits = torch.ones(l_features_un.shape[0], self.args.num_classes, dtype=torch.long).to(self.args.device)
        for i in range(self.args.num_classes):
            unique_indices, counts = torch.unique(top_k_index[np.asarray(g_labelArr_all)==i], return_counts=True)
            l_unique_indices, l_counts = torch.unique(top_k_index[np.asarray(l_labelArr_all)==i], return_counts=True)
            rknn_logits[unique_indices,i] += counts
            l_rknn_logits[l_unique_indices,i] += l_counts
        rknn_logits = rknn_logits.cpu()
        l_rknn_logits = l_rknn_logits.cpu()
        l_probs1 = rknn_logits/(rknn_logits.sum(1).reshape(-1,1))
        l_probs2 = l_rknn_logits/(l_rknn_logits.sum(1).reshape(-1,1))
        l_pred1 = l_probs1.max(1)[1]
        l_pred2 = l_probs2.max(1)[1]
        l_c_pred = l_c_probs.max(1)[1].cpu()
        
        l_kd_labelArr_all = self.get_rknn_label(label_idxs, unlabel_idxs, l_net_kd)

        l_kd_features_l = self.get_embedding(label_idxs, l_net_kd, GPU=True)
        l_kd_features_l = F.normalize(l_kd_features_l, dim=1)
        l_kd_features_un, l_kd_c_probs = self.get_embedding_and_prob(unlabel_idxs, l_net_kd, GPU=True)
        l_kd_features_un = F.normalize(l_kd_features_un, dim=1)
        dists_all = torch.mm(torch.vstack((l_kd_features_l,l_kd_features_un)), l_kd_features_un.t())
        dists_all[torch.arange(l_kd_features_l.size()[0], dists_all.size()[0]), torch.arange(l_kd_features_un.size()[0])] = -1
        _, top_k_index = dists_all.topk(250, dim=1, largest=True, sorted=True) ## Top-K similar scores and corresponding indexes
        rknn_logits = torch.ones(l_kd_features_un.shape[0], self.args.num_classes, dtype=torch.long).to(self.args.device)
        l_kd_rknn_logits = torch.ones(l_kd_features_un.shape[0], self.args.num_classes, dtype=torch.long).to(self.args.device)
        for i in range(self.args.num_classes):
            unique_indices, counts = torch.unique(top_k_index[np.asarray(g_labelArr_all)==i], return_counts=True)
            l_kd_unique_indices, l_kd_counts = torch.unique(top_k_index[np.asarray(l_kd_labelArr_all)==i], return_counts=True)
            rknn_logits[unique_indices,i] += counts
            l_kd_rknn_logits[l_kd_unique_indices,i] += l_kd_counts
        rknn_logits = rknn_logits.cpu()
        l_kd_rknn_logits = l_kd_rknn_logits.cpu()
        l_kd_probs1 = rknn_logits/(rknn_logits.sum(1).reshape(-1,1))
        l_kd_probs2 = l_kd_rknn_logits/(l_kd_rknn_logits.sum(1).reshape(-1,1))
        l_kd_pred1 = l_kd_probs1.max(1)[1]
        l_kd_pred2 = l_kd_probs2.max(1)[1]
        l_kd_c_pred = l_kd_c_probs.max(1)[1].cpu()

        D_ll = torch.tensor([wasserstein_distance(l_probs1[i], l_kd_probs1[i]) for i in range(l_probs1.shape[0])])
        D_gl = torch.tensor([wasserstein_distance(g_probs1[i], l_kd_probs1[i]) for i in range(l_probs1.shape[0])])
        print(torch.max(D_gl/(D_ll+1e-5), D_ll/(D_gl+1e-5)), D_ll, D_gl)
        U1 = torch.max(D_gl/(D_ll+1e-5), D_ll/(D_gl+1e-5))*(D_ll + D_gl)
        U1 = U1.numpy()
        
        U2 = torch.tensor([wasserstein_distance(l_probs2[i], l_c_probs.cpu()[i]) for i in range(l_probs2.shape[0])])
        U2 = U2.numpy()
        
        U = U1*U2
        
        th = threshold_otsu(U.reshape(-1,1))
        print("First time th: ", th)

        print("Global model prediction accuracy: ", (g_pred2==g_c_pred).sum()/g_c_pred.shape[0])
        print("Local model prediction accuracy: ", (l_pred2==l_c_pred).sum()/l_c_pred.shape[0])
        print("Local KD model prediction accuracy: ", (l_kd_pred2==l_kd_c_pred).sum()/l_kd_c_pred.shape[0])
        
        l_features_un = l_features_un.cpu()
        
        candidate_l_features_un = l_features_un[U>th]
        candidate_unlabel_idxs = unlabel_idxs[U>th]
        
        if len(candidate_unlabel_idxs) > 0:
            print("len(candidate_unlabel_idxs) :", len(candidate_unlabel_idxs))
            selected_idxs = self.sample(min(n_query, len(candidate_unlabel_idxs)), feats=candidate_l_features_un)
            selected_idxs = candidate_unlabel_idxs[selected_idxs]
        else:
            selected_idxs = np.array([], dtype=np.int)
        
        if len(selected_idxs) < n_query:
            U = deepcopy(U1)
            th = threshold_otsu(U.reshape(-1,1))
            print("Second time th: ", th)
            
            candidate_l_features_un = l_features_un[U>th]
            candidate_unlabel_idxs = unlabel_idxs[U>th]
            
            if len(candidate_unlabel_idxs) > 0:
                print("again len(candidate_unlabel_idxs) :", len(candidate_unlabel_idxs))
                selected_idxs = self.sample(min(n_query, len(candidate_unlabel_idxs)), feats=candidate_l_features_un)
                selected_idxs = candidate_unlabel_idxs[selected_idxs]
            else:
                selected_idxs = np.array([], dtype=np.int)
            
            if len(selected_idxs) < n_query:
                return unlabel_idxs[(U1*U2).argsort()[-n_query:]]
            
        return np.array(selected_idxs)
    
    def sample(self, n, feats):
        feats = feats.numpy()
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(feats)

        cluster_idxs = cluster_learner.predict(feats)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (feats - centers) ** 2
        dis = dis.sum(axis=1)
        return np.array(
            [np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if
                (cluster_idxs == i).sum() > 0])
