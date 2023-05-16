import numpy as np
import torch
import argparse
from sklearn.model_selection import KFold
import torch.nn.functional as F
import random


EPS = 1e-10

# def load_data():
def count_correct(output, target):
    pred = output.max(1)[1].type_as(target)
    correct = pred.eq(target).sum().item()
    return correct

def attn_loss(a, r):
    a = a.sort(dim=1).values
    loss = -torch.log(a[:,-int(a.size(1)*r):]+EPS).mean() -torch.log(1-a[:,:int(a.size(1)*r)]+EPS).mean()
    return loss

def consist_loss(s, device):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0],s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    L = D-W
    L = L.to(device)
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res

def train_val_test_split(kfold = 5, fold = 0, seed=123):
    n_sub = 1035
    id = list(range(n_sub))


    random.seed(seed)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=seed,shuffle = True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr,te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]


    return train_id,val_id,test_id


# we dont need k-fold here, get test_set by mod 3
def train_test_split(num_aug=3):
    N = 1035*num_aug
    train_to = int(0.6*N)
    val_to = int(0.8*N)
    print(train_to)
    print(val_to)
    id = list(range(N))

    random.shuffle(id)

    train_id = id[:train_to]
    val_id = id[train_to:val_to]
    test_id = id[val_to:]
    test_idx = [x for x in test_id if x%num_aug==0]

    return train_id, val_id, test_idx

