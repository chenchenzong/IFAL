import copy
import numpy as np
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_wts = model.state_dict()
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.best_model_wts = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss has not improved for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_wts)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label, item
    
    
class Strategy:
    def __init__(self, dataset_query, dataset_train, net, args):
        self.dataset_query = dataset_query
        self.dataset_train = dataset_train
        self.net = net
        self.args = args
        self.local_net_dict = {}
        self.loss_func = nn.CrossEntropyLoss()
        
    def query(self, label_idx, unlabel_idx):
        pass
    
    def get_label_count(self, label_idxs):
        loader_te = DataLoader(DatasetSplit(self.dataset_query, label_idxs), shuffle=False)
        
        label_count = [0 for i in range(self.args.num_classes)]
        with torch.no_grad():
            for x, y, idxs in loader_te:
                label_count[y[0].item()] += 1

        return label_count
    
    def get_rknn_label(self, label_idxs, unlabel_idxs, net=None):
        loader_te = DataLoader(DatasetSplit(self.dataset_query, label_idxs), shuffle=False)
        
        if net is None:
            net = self.net
        
        all_labels = []
        with torch.no_grad():
            for x, y, idxs in loader_te:
                all_labels.append(y[0].item())

        loader_te = DataLoader(DatasetSplit(self.dataset_query, unlabel_idxs), shuffle=False)
        
        net.eval()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                output, emb = net(x)
                pred = output.max(1)[1]
                all_labels.append(pred[0].item())
        return all_labels
    
    def get_embedding_and_prob(self, data_idxs, net=None, GPU=False):
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        if net is None:
            net = self.net
        
        net.eval()
        embedding = torch.zeros([len(data_idxs), net.get_embedding_dim()])
        probs = torch.zeros([len(data_idxs), self.args.num_classes])
        if GPU:
            embedding = embedding.to(self.args.device)
            probs = probs.to(self.args.device)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                out, e1 = net(x)
                if GPU:
                    embedding[idxs] = e1.data
                    probs[idxs] = torch.nn.functional.softmax(out, dim=1).data
                else:
                    embedding[idxs] = e1.data.cpu()
                    probs[idxs] = torch.nn.functional.softmax(out, dim=1).cpu().data
        
        return embedding, probs
    
    def predict_prob(self, unlabel_idxs, net=None, GPU=False):
        loader_te = DataLoader(DatasetSplit(self.dataset_query, unlabel_idxs), shuffle=False)
        
        if net is None:
            net = self.net
            
        net.eval()
        probs = torch.zeros([len(unlabel_idxs), self.args.num_classes])
        if GPU:
            probs = probs.to(self.args.device)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                output, emb = net(x)
                if GPU:
                    probs[idxs] = torch.nn.functional.softmax(output, dim=1).data
                else:
                    probs[idxs] = torch.nn.functional.softmax(output, dim=1).cpu().data
        return probs


    def get_embedding(self, data_idxs, net=None, GPU=False):
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        if net is None:
            net = self.net
        
        net.eval()
        embedding = torch.zeros([len(data_idxs), net.get_embedding_dim()])
        if GPU:
            embedding = embedding.to(self.args.device)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                out, e1 = net(x)
                if GPU:
                    embedding[idxs] = e1.data
                else:
                    embedding[idxs] = e1.data.cpu()
        
        return embedding
    
    
    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, data_idxs, net=None):
        if net is None:
            net = self.net
            
        embDim = net.get_embedding_dim()
        net.eval()
        
        nLab = self.args.num_classes 
        embedding = np.zeros([len(data_idxs), embDim * nLab])
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                cout, out = net(x)
                out = out.data.cpu().numpy()
                
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)
        
        
    def get_grad_embedding_maxInd(self, data_idxs, net=None):
        if net is None:
            net = self.net
            
        embDim = net.get_embedding_dim()
        net.eval()
        
        nLab = self.args.num_classes 
        embedding = np.zeros([len(data_idxs), embDim])
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                cout, out = net(x)
                out = out.data.cpu().numpy()
                
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                            
            return torch.Tensor(embedding)
    
    
    def training_local_only(self, label_idxs, finetune=False):
        finetune_ep = 50
        
        local_net = deepcopy(self.net)
        if not finetune: 
            # Training Local Model from the scratch
            local_net.load_state_dict(self.args.raw_ckpt)
        # else: fine-tune from global model checkpoint
        
        # train and update
        label_train = DataLoader(DatasetSplit(self.dataset_train, label_idxs), batch_size=self.args.local_bs, shuffle=True)
        
        optimizer = torch.optim.SGD(local_net.parameters(), 
                                    lr=self.args.lr, 
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(finetune_ep * 3 / 4)], gamma=self.args.lr_decay)
        
        # start = datetime.now()
        for epoch in range(finetune_ep):
            local_net.train()
            for images, labels, _ in label_train:
                if self.args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
                    labels = labels.squeeze().long()
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                output, emb = local_net(images)
                
                if output.shape[0] == 1:
                    labels = labels.reshape(1,)

                loss = self.loss_func(output, labels)
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
            correct, cnt = 0., 0.
            local_net.eval()
            with torch.no_grad():
                for images, labels, _ in label_train:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    output, _ = local_net(images)
                    
                    y_pred = output.data.max(1, keepdim=True)[1]
                    correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                    cnt += len(labels)
        
                acc = correct / cnt
                if acc >= 0.99:
                    break
        
        # time = datetime.now() - start
        # print('Local-only model fine-tuning takes {}'.format(time))

        return local_net


    def training_local_only_KD(self, label_idxs, finetune=False):
        finetune_ep = 500
        
        global_net = deepcopy(self.net)
        local_net = deepcopy(self.net)
        if not finetune: 
            # Training Local Model from the scratch
            local_net.load_state_dict(self.args.raw_ckpt)
        # else: fine-tune from global model checkpoint
        
        # train and update
        label_train = DataLoader(DatasetSplit(self.dataset_train, label_idxs), batch_size=self.args.local_bs, shuffle=True)
        
        optimizer = torch.optim.SGD(local_net.parameters(), 
                                    lr=self.args.lr*0.1, 
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        early_stopping = EarlyStopping(patience=5)
        
        # start = datetime.now()
        for epoch in range(finetune_ep):
            local_net.train()
            global_net.eval()
            running_loss = 0.0
            for images, labels, _ in label_train:
                if self.args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
                    labels = labels.squeeze().long()
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                output, emb = local_net(images)
                global_output, _ = global_net(images)
                
                if output.shape[0] == 1:
                    labels = labels.reshape(1,)

                loss_ce = self.loss_func(output, labels)
                loss_kd = kd_loss(output, global_output, 4.)
                loss = 0.1*loss_ce + 0.9*loss_kd
                loss.backward()
                
                optimizer.step()
                running_loss += loss.item()
            
            running_loss /= len(label_train)
            early_stopping(running_loss, local_net)
            
            if early_stopping.early_stop:
                print("Early stopping triggered, ", epoch)
                early_stopping.load_best_model(local_net)
                break

        return local_net