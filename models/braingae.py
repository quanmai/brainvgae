import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import time
import torch.optim as optim
import os
import glob
import time

from torch_geometric.nn import GATConv, TopKPooling, GINConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_adj, dense_to_sparse, remove_self_loops
import torch_geometric.transforms as T

EPS = 1E-10
EDGE = 20
BEST = 1000
WARM_UP = 50    
DIM = 64
REVERSE = 0.8

gdc = T.GDC(self_loop_weight=None, 
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=20, dim=0), 
            exact=True,
            )

class BrainGAE(object):
    def __init__(self, in_channels, out_channels, num_site, ratio, alpha, n_epochs, lr, weight_decay, temperature, site_adapt, augment, train_loader, val_loader, test_loader, threshold, l0, l1, l2, l3, pretrain_cl, pretrain_ep, beta, sampling, mix, aggr, gnn, edgepredictor):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BrainGAE_model(in_channels,
                                    out_channels,
                                    num_site,
                                    ratio,
                                    alpha,
                                    temperature,
                                    site_adapt,
                                    sampling,
                                    aggr,
                                    gnn,
                                    edgepredictor)#.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.optimizer, self.scheduler = self._opt()
        self.augment = augment
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.l0 = l0 # class_loss
        if site_adapt:
            self.l1 = l1 # site_loss
        else:
            self.l1 = 0
        self.l2 = l2 # attn_loss
        self.l3 = l3 # ep_loss
        self.ratio = ratio # node drop prob
        self.site_adapt = site_adapt
        self.num_nodes = in_channels
        self.threshold = threshold
        self.pretrain_cl = pretrain_cl
        self.pretrain_ep = pretrain_ep
        self.beta = beta
        self.mix = mix
        # self.sampling = sampling
    def _opt(self):
        import torch.optim
        optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay
                                    )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=20,
                                                    gamma=0.5
                                                    )
        return optimizer, scheduler

    def pretrain_ep_net(self):
        print('Pretraining ep net--')
        optimizer = torch.optim.Adam(self.model.ep.parameters(),
                                     lr=self.lr/2
                                     )
        for _ in range(100):    
            self.model.train()
            for data in self.train_loader:
                data.to(self.device)
                adj_logits = self.model.ep(data.x, data.edge_index)
                adj_org_dense = to_dense_adj(data.edge_index, data.batch, data.edge_attr).squeeze()
                m = self.model.ep.m
                s = self.model.ep.s
                kl_loss = -0.5*torch.mean(torch.sum(1+2*s-m**2-s.exp()**2, dim=1))
                loss = F.binary_cross_entropy_with_logits(adj_logits, adj_org_dense, pos_weight=None)
                # m, s = self.model.ep.encoder(data.x, data.edge_index)
                # loss = self.model.ep.kl_loss(m, s)
                loss -= kl_loss
                # loss = -loss
                loss.backward()
                optimizer.step()
        print('--Done pretraining ep net')
    
    def pretrain_classifier(self):
        print('Pretraining nc net--')
        optimizer = torch.optim.Adam(self.model.classifier.parameters(),
                                     lr=self.lr,
                                     )
        for _ in range(10):    
            self.model.train()
            for data in self.train_loader:
                data.to(self.device)
                output = self.model.classifier(data, data.edge_index, data.edge_attr)
                loss = F.nll_loss(output[0], data.y)
                if self.site_adapt:
                    loss += self.l1*F.nll_loss(output[1], data.site)
                loss.backward()
                optimizer.step()
        print('--Done pretraining nc net')

    def train(self, model, epoch):
        t = time.time()
        correct_train = 0
        loss_train = 0
        if self.mix:
            adj_lst = []
        model.train()

        for data in self.train_loader:
            data.to(self.device)
            self.optimizer.zero_grad()
            if self.site_adapt:
                label, site, a1, a2, adj_org, adj_logits = model(data, data.edge_index, None)
                loss_st = F.nll_loss(site, data.site)
                loss_st_nonreduce = F.nll_loss(site, data.site, reduction='none') + EPS
                loss_st_nonreduce = torch.reshape(loss_st_nonreduce, (data.num_graphs, 1, 1))
            else:
                label, a1, a2, adj_org, adj_logits = model(data, data.edge_index, None)
                loss_st = 0
            loss_ep = F.binary_cross_entropy_with_logits(adj_org, adj_logits.squeeze(), pos_weight=None)
            if self.mix:
                loss_nonreduce = F.nll_loss(label, data.y, reduction='none') + EPS
                loss_nonreduce = torch.reshape(loss_nonreduce, (data.num_graphs, 1, 1))
                if self.site_adapt:
                    loss_nonreduce += self.l1*loss_st_nonreduce
                adj_logits_norm = adj_logits/torch.max(adj_logits)
                adj_lst.append(torch.mean(torch.div(adj_logits_norm, torch.exp(loss_nonreduce)), 0, keepdim=False))
            loss_cl = F.nll_loss(label, data.y)
            loss_at = self.attn_loss(a1, self.ratio) + self.attn_loss(a2, self.ratio)
            loss = self.l0*loss_cl + self.l1*loss_st + self.l2*loss_at + self.l3*loss_ep
            correct_train += self.count_correct(label, data.y)
            loss_train += loss_cl.item() * data.num_graphs
            loss.backward()
            self.optimizer.step()
            # del data, loss_nonreduce, adj_org, adj_logits, label, adj_logits_norm
            # torch.cuda.empty_cache()
        self.scheduler.step()

        acc_train = correct_train / len(self.train_loader.dataset)
        loss_train /= len(self.train_loader.dataset)
        if self.mix:
            adj_train = torch.mean(torch.stack(adj_lst), 0, keepdim=False)
            adj_train = torch.div(adj_train, torch.max(adj_train, 0)[0])
            # adj_threshold = (adj_train >= 0.58).type_as(adj_train)
            adj_threshold = self.topk(adj_train, self.threshold)

            adj_sparse, _ = dense_to_sparse(adj_threshold)
            adj_sparse.to(self.device)
            adj_threshold.to(self.device)
        
        loss_val = 0
        correct_val = 0
            
        model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                data.to(self.device)
                if self.mix:
                    edge_dense = to_dense_adj(data.edge_index, data.batch, data.edge_attr).squeeze()
                    adj = (1-self.beta)*adj_threshold + self.beta*edge_dense
                    adj = self.topk(adj, self.threshold)
                    adj, _ = dense_to_sparse(adj)
                    adj, _ = remove_self_loops(adj)
                    output = model.classifier(data,adj,None)
                else: 
                    edge_index, _ = remove_self_loops(data.edge_index)
                    output = model.classifier(data,edge_index,None)
                loss_cl = F.nll_loss(output[0], data.y)
                if self.site_adapt:
                    loss_cl += self.l1*F.nll_loss(output[1], data.site)
                loss_val += loss_cl.item() * data.num_graphs
                correct_val += self.count_correct(output[0], data.y)
                del data, output
                torch.cuda.empty_cache()
            acc_val = correct_val / len(self.val_loader.dataset)
            loss_val /= len(self.val_loader.dataset)
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train),
            'acc_train: {:.4f}'.format(acc_train),
            'loss_val: {:.4f}'.format(loss_val),
            'acc_val: {:.4f}'.format(acc_val),
            'time: {:.4f}s'.format(time.time() - t)
        )
        # del adj_threshold
        # torch.cuda.empty_cache()
        if self.mix:
            return loss_val, adj_threshold# adj_sparse
        return loss_val

    def fit(self, patience):
        t_total = time.time()
        if self.mix:
            adj_best = torch.zeros([200,200], dtype=torch.float)
        loss_values = []
        bad_count = 0
        best = BEST
        best_epoch = 0
        model = self.model.to(self.device)
        if self.pretrain_ep:
            self.pretrain_ep_net()
        if self.pretrain_cl:
            self.pretrain_classifier()
        for epoch in range(self.n_epochs):
            if self.mix:
                loss_v, adj = self.train(model, epoch)
            else:
                loss_v = self.train(model, epoch)
            loss_values.append(loss_v)
            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
            if epoch > WARM_UP:
                if loss_values[-1] < best:
                    best = loss_values[-1]
                    if self.mix:
                        adj_best = adj
                    best_epoch = epoch
                    bad_count = 0
                else:
                    bad_count += 1
                    if self.mix:
                        del adj
                        torch.cuda.empty_cache()
                if bad_count == patience: 
                    break
                
                files = glob.glob('*.pkl')
                for file in files:
                    epoch_nb = int(file.split('.')[0])
                    if epoch_nb < best_epoch:
                        os.remove(file)

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)
                
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore the best model
        print('Loading {}th epoch'.format(best_epoch))
        model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

        # Testing
        if self.mix:
            self.test(model, adj_best)
        else:
            self.test(model)
    
    @torch.no_grad()
    def test(self, model, adj_threshold=None):
        correct_test, loss_test = 0, 0
        model.eval()
        if adj_threshold is not None:
            adj_threshold.to(self.device)
        for data in self.test_loader:
            data.to(self.device)
            if self.mix:
                edge_dense = to_dense_adj(data.edge_index, data.batch, data.edge_attr).squeeze()
                adj = (1-self.beta)*adj_threshold + self.beta*edge_dense
                adj = self.topk(adj, self.threshold)
                adj, _ = dense_to_sparse(adj)
                adj, _ = remove_self_loops(adj)
                output = model.classifier(data, adj, None)
            else:
                edge_index, _ = remove_self_loops(data.edge_index)
                output = model.classifier(data,edge_index,None)
            loss = F.nll_loss(output[0], data.y)
            loss_test += loss.item() * data.num_graphs
            correct_test += self.count_correct(output[0], data.y)
        acc_test = correct_test / len(self.test_loader.dataset)
        loss_test = loss_test / len(self.test_loader.dataset)
        print("Test set result:",
            "loss= {:.4f}".format(loss_test),
            "accuracy= {:.4f}".format(acc_test))

    def topk(self, A, K=None):
        if K is None:
            K=self.threshold
        num_nodes = A.shape[1]
        row_index = np.arange(num_nodes)
        k = int(num_nodes*K)
        A[torch.argsort(A, dim=0)[:num_nodes-k],row_index] = 0.0
        A = (A > 0.0).type_as(A)
        # A = A.triu(0) + torch.transpose(A.triu(1), 0, 1)
        return A

    @staticmethod
    def attn_loss(a, r):
        a = a.sort(dim=1).values
        loss = -torch.log(a[:,-int(a.size(1)*r):]+EPS).mean() -torch.log(1-a[:,:int(a.size(1)*r)]+EPS).mean()
        return loss

    @staticmethod
    def count_correct(output, target):
        pred = output.max(1)[1].type_as(target)
        correct = pred.eq(target).sum().item()
        return correct


class BrainGAE_model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_site, ratio, alpha, temperature, site_adapt, sampling, aggr,gnn,edgepredictor):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        # self.ep = VGAE(EdgePredictor(in_channels))
        self.ep = VGAE(in_channels)
        self.classifier = GraphClassifier(in_channels, out_channels, num_site, ratio, site_adapt, aggr,gnn)
        self.site_adapt = site_adapt
        self.sampling = sampling
        self.edgepredictor = edgepredictor

    def bernouliSampling(self, adj_logits, adj_org):
        adj_logits = F.sigmoid(adj_logits)
        edge_probs = adj_logits / torch.max(adj_logits)
        
        edge_probs = self.alpha*edge_probs + (1-self.alpha)*adj_org
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        adj_sampled = adj_sampled.triu(0) + torch.transpose(adj_sampled.triu(1), 1, 2)

        return adj_sampled, dense_to_sparse(adj_sampled)[0], dense_to_sparse(adj_sampled)[1]

    def add_remove_adj(self, adj_logits, adj_org, theta=0.2):
        batch_size, num_nodes = adj_logits.shape[0], adj_logits.shape[1]
        # print(batch_size)
        # normalize adj_logits 
        edge_probs = adj_logits
        edge_probs = edge_probs - torch.min(torch.min(edge_probs, dim=1).values,dim=-1).values.reshape(-1,1,1)
        edge_probs = edge_probs / torch.max(torch.max(edge_probs, dim=1).values,dim=-1).values.reshape(-1,1,1)
        
        n_edges = int(torch.nonzero(adj_org).shape[0]/batch_size)
        n_changes = int(EDGE * theta/2)

        adj_inv = 1-adj_org
        # element-wise product
        mask_rm = edge_probs*adj_org
        mask_rm_lst = []
        # row_index = np.arange(num_nodes)
        for i in range(batch_size):
            mask_rm_i = mask_rm[i]
            thres_rm = torch.topk(mask_rm_i, n_changes, dim=0, largest=True)[0][-1]
            mask_rm_i[mask_rm_i<thres_rm] = 0
            mask_rm_i = CeilNoGradient.apply(mask_rm_i)
            mask_rm_lst.append(mask_rm_i)
        mask_rm = torch.stack(mask_rm_lst)
        adj_new = adj_org - mask_rm

        mask_add = edge_probs*adj_inv
        mask_add_lst = []
        for i in range(batch_size):
            mask_add_i = mask_add[i]
            thres_add = torch.topk(mask_add_i, n_changes, dim=0, largest=True)[0][-1]
            mask_add_i[mask_add_i<thres_add] = 0
            mask_add_i = CeilNoGradient.apply(mask_add_i)
            mask_add_lst.append(mask_add_i)
        mask_add = torch.stack(mask_add_lst)
        adj_new = adj_new + mask_add

        return adj_new, dense_to_sparse(adj_new)[0], dense_to_sparse(adj_new)[1]

    def forward(self, data, edge_index, edge_attr):
        x, batch = data.x, data.batch

        adj_logits = self.ep(x, edge_index) #dense
        edge_dense_org = self.to_dense_(edge_index, batch, edge_attr)
        if self.edgepredictor:
            if self.sampling=='bernouli':
                _, edge_sparse_sampled, edge_sparse_attr = self.bernouliSampling(adj_logits, edge_dense_org)
            else:
                _, edge_sparse_sampled, edge_sparse_attr = self.add_remove_adj(adj_logits, edge_dense_org)
        else:
            edge_sparse_sampled, edge_sparse_attr = edge_index, edge_attr
        if not self.site_adapt:
            # xy, attn1_sig, attn2_sig = self.classifier(data, edge_index, edge_attr)
            xy, attn1_sig, attn2_sig = self.classifier(data, edge_sparse_sampled, edge_sparse_attr)
            return xy, attn1_sig, attn2_sig, edge_dense_org, adj_logits
        else:
            # xy, xs, attn1_sig, attn2_sig = self.classifier(data, edge_index, edge_attr)
            xy, xs, attn1_sig, attn2_sig = self.classifier(data, edge_sparse_sampled, edge_sparse_attr)
            return xy, xs, attn1_sig, attn2_sig, edge_dense_org, adj_logits

    def renormalization(self):
        pass

    @staticmethod
    def to_dense_(edge_index, batch, edge_attr):
        edge_dense = to_dense_adj(edge_index, batch, edge_attr).squeeze()
        edge_dense = (edge_dense > torch.zeros_like(edge_dense)).type_as(edge_dense)
        return edge_dense

class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_site, ratio, site_adapt, aggr='cat', gnn='gcn'):
        super().__init__()
        dim = DIM
        if gnn=='gcn':
            self.conv1 = GCNConv(in_channels, dim, cached=False, normalize=False)
            self.pool1 = TopKPooling(dim, ratio=ratio)
            self.conv2 = GCNConv(dim, dim, cached=False, normalize=False)
            self.pool2 = TopKPooling(dim, ratio=ratio)
        else: #GIN
            self.conv1 = GINConv(nn.Sequential(nn.Linear(in_channels, dim), nn.ReLU(), nn.Linear(dim, dim)))
            self.pool1 = TopKPooling(dim, ratio=ratio)
            self.conv2 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
            self.pool2 = TopKPooling(dim, ratio=ratio)
        # self.conv3 = GCNConv(dim, dim, cached=False, normalize=False)
        # self.pool3 = TopKPooling(dim, ratio=ratio)
        if aggr == 'cat' or aggr == 'catmeanmanx':
            dim = 2*DIM
        self.fc1 = nn.Linear(2*dim, 4*dim) #2dim, 4dim
        self.bn1 = nn.BatchNorm1d(4*dim)
        self.fc2 = nn.Linear(4*dim, 8*dim)
        self.bn2 = nn.BatchNorm1d(8*dim)
        self.fc3 = nn.Linear(8*dim, out_channels)

        if site_adapt:
            self.siteclassifier = site_classifier(2*dim, num_site)
        else:
            self.siteclassifier = None
        self.aggr = aggr

    def forward(self, data, edge_index, edge_attr):
        x, batch = data.x, data.batch
        x = F.relu(self.conv1(x, edge_index)) #, edge_weight=None))
        x, edge_index, edge_attr, batch, _, attn1 =self.pool1(x, edge_index, edge_attr, batch) #(160, 64))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) #(1,2dim)

        x = F.relu(self.conv2(x, edge_index)) #, edge_weight=None))
        x, edge_index, edge_attr, batch, _, attn2 = self.pool2(x, edge_index, edge_attr, batch) #(160, 64))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) #(1, 2dim)

        # x = F.relu(self.conv3(x, edge_index, edge_weight=None))
        # x, edge_index, edge_attr, batch, _, attn3 = self.pool3(x, edge_index, edge_attr, batch) #(160, 64))
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) #(1, 2dim)
        if self.aggr=='cat':
            x = torch.cat([x1,x2],dim=1)
        elif self.aggr=='max':
            x = torch.maximum(x1,x2)
        elif self.aggr=='catmeanmanx':
            x_max = torch.maximum(x1,x2)
            x_mean = (x1+x2)/2
            x = torch.cat([x_max,x_mean],dim=1)
        # elif self.aggr == 'mean':
        #     x = torch.mean(x1,x2)
        else:
            x = x1 + x2 
        xy = self.bn1(F.relu(self.fc1(x)))
        xy = F.dropout(xy, p=0.6, training=self.training)
        xy = self.bn2(F.relu(self.fc2(xy)))
        xy = F.log_softmax(self.fc3(xy), dim=-1)

        if self.siteclassifier is not None: 
            xs = self.siteclassifier(x)
            return xy, xs, torch.sigmoid(attn1.view(x.size(0),-1)), torch.sigmoid(attn2.view(x.size(0),-1))
        return xy, torch.sigmoid(attn1.view(x.size(0),-1)), torch.sigmoid(attn2.view(x.size(0),-1))

class EdgePredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        embedded_channels = in_channels
        self.conv_base = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, embedded_channels)
        self.conv_logstd = GCNConv(hidden_channels, embedded_channels)

    def forward(self, x, edge_index):
        h = self.conv_base(x, edge_index)
        m = self.conv_mu(h, edge_index)#.relu()
        s = self.conv_logstd(h, edge_index)#.relu()
        return m, s

class VGAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.nodes = in_channels
        embedded_channels = in_channels
        self.conv_base = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, embedded_channels)
        self.conv_logstd = GCNConv(hidden_channels, embedded_channels)

    def forward(self, x, adj):
        h = self.conv_base(x, adj)
        self.m = self.conv_mu(h, adj).relu()
        self.s = self.conv_logstd(h, adj).relu()
        z = self.m + torch.randn_like(self.s) * torch.exp(self.s)
        z = z.view(-1, self.nodes, self.nodes)
        adj_logits = torch.matmul(z, torch.transpose(z, 1, 2))
        return adj_logits

def get_topk_matrix(self, A, K=0.1):
    num_nodes = A.shape[1]
    row_index = np.arange(num_nodes)
    k = int(num_nodes*K)
    A[torch.argsort(A, dim=0)[:num_nodes-k],row_index] = 0.0
    A = (A > 0.0).type_as(A)
    # A = A.triu(0) + torch.transpose(A.triu(1), 0, 1)
    return A

class CeilNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()

    @staticmethod
    def backward(ctx, g):
        return g

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return REVERSE*grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

class site_classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(site_classifier, self).__init__()
        # self.fc1 = nn.Linear(in_features, 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, out_features)
        # self.bn1 = nn.BatchNorm1d(256)
        # self.bn2 = nn.BatchNorm1d(256)
        dim = int(0.5*in_features)
        self.fc1 = nn.Linear(in_features, dim) #2dim, 4dim
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, 2*dim)
        self.bn2 = nn.BatchNorm1d(2*dim)
        self.fc3 = nn.Linear(2*dim, out_features)
    def forward(self, x):
        x = grad_reverse(x)
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x