from locale import normalize
from typing import ForwardRef
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import time
import torch.optim as optim

from torch_geometric.nn import GATConv, TopKPooling, GINConv, GCNConv, VGAE
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from utils.layers import GraphAttentionLayer, site_classifier

class pyGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(pyGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout = dropout, alpha = alpha, concat = True) 
                            for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_site, ratio, site_adapt=True):
        super().__init__()
        self.num_site = num_site
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, 64, heads=1, concat=False,
                             dropout=0.6)
        self.pool1 = TopKPooling(64, ratio=ratio)
        self.pool2 = TopKPooling(64, ratio=ratio)
        self.pool2 = TopKPooling(64, ratio=ratio)
        self.fc1 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, out_channels)
        if site_adapt:
            self.siteclassifier = site_classifier(128, self.num_site)
        else:
            self.siteclassifier = None

    def forward(self, data):
        # attn = torch.softmax(data.x, dim=0).flatten()
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index)) #(200,64)
        x, edge_index, edge_attr, batch, _, attn1 = self.pool1(x, edge_index, edge_attr, batch) #(160, 64))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) #(1,128)
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index) #(160,16)
        x, edge_index, edge_attr, batch, _, attn2 = self.pool2(x, edge_index, edge_attr, batch) #(128,64)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) #(1, 128)

        # x = F.dropout(x, p=0.6, training=self.training)
        # x = self.conv
        x = x1 + x2 # Final feature extraction
        
        
        xy = self.bn1(F.relu(self.fc1(x)))
        xy = F.dropout(xy, p=0.6, training=self.training)
        xy = self.bn2(F.relu(self.fc2(xy)))
        xy = F.log_softmax(self.fc3(xy), dim=-1)
        
        if self.siteclassifier is not None: 
            xs = self.siteclassifier(x)
            return xy, xs, torch.sigmoid(attn1.view(x.size(0),-1)), torch.sigmoid(attn2.view(x.size(0),-1))
        return xy, torch.sigmoid(attn1.view(x.size(0),-1)), torch.sigmoid(attn2.view(x.size(0),-1))

class GIN(nn.Module):
    def __init__(self, in_channels, out_channels, num_site, ratio, site_adapt=True):
        super().__init__()
        dim = 32
        self.num_site = num_site
        self.conv1 = GINConv(nn.Sequential(nn.Linear(in_channels, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.pool1 = TopKPooling(dim, ratio=ratio)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.pool2 = TopKPooling(dim, ratio=ratio)
        self.fc1 = nn.Linear(2*dim, 4*dim)
        self.bn1 = nn.BatchNorm1d(4*dim)
        self.fc2 = nn.Linear(4*dim, 8*dim)
        self.bn2 = nn.BatchNorm1d(8*dim)
        self.fc3 = nn.Linear(8*dim, out_channels)
        self.fc4 = nn.Linear(4*dim, out_channels)
        if site_adapt:
            self.siteclassifier = site_classifier(2*dim, self.num_site)
        else:
            self.siteclassifier = None

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, attn1 = self.pool1(x, edge_index, edge_attr, batch) #(160, 64))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) #(1,128)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, attn2 = self.pool2(x, edge_index, edge_attr, batch) #(160, 64))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) #(1, 128)

        x = x1 + x2

        xy = self.bn1(F.relu(self.fc1(x)))
        xy = F.dropout(xy, p=0.6, training=self.training)
        xy = self.bn2(F.relu(self.fc2(xy)))
        xy = F.log_softmax(self.fc3(xy), dim=-1)

        # xy = F.log_softmax(self.fc4(xy), dim=-1)
        
        if self.siteclassifier is not None: 
            xs = self.siteclassifier(x)
            return xy, xs, torch.sigmoid(attn1.view(x.size(0),-1)), torch.sigmoid(attn2.view(x.size(0),-1))
        return xy, torch.sigmoid(attn1.view(x.size(0),-1)), torch.sigmoid(attn2.view(x.size(0),-1))

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_site, ratio, site_adapt=True):
        super().__init__()
        dim = 16
        self.conv1 = GCNConv(in_channels, dim, cached=True, normalize=False)
        self.pool1 = TopKPooling(dim, ratio=ratio)
        self.conv2 = GCNConv(dim, dim, cached=True, normalize=False)
        self.pool2 = TopKPooling(dim, ratio=ratio)
        self.fc1 = nn.Linear(2*dim, 4*dim) #2dim, 4dim
        self.bn1 = nn.BatchNorm1d(4*dim)
        self.fc2 = nn.Linear(4*dim, 8*dim)
        self.bn2 = nn.BatchNorm1d(8*dim)
        self.fc3 = nn.Linear(8*dim, out_channels)

        if site_adapt:
            self.siteclassifier = site_classifier(2*dim, num_site)
        else:
            self.siteclassifier = None

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight=None))
        x, edge_index, edge_attr, batch, _, attn1 = self.pool1(x, edge_index, edge_attr, batch) #(160, 64))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) #(1,2dim)

        x = F.relu(self.conv2(x, edge_index, edge_weight=None))
        x, edge_index, edge_attr, batch, _, attn2 = self.pool2(x, edge_index, edge_attr, batch) #(160, 64))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) #(1, 2dim)

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
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        embedded_channels = in_channels
        self.conv_base = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, embedded_channels)
        self.conv_logstd = GCNConv(hidden_channels, embedded_channels)

    def forward(self, x, edge_index):
        h = self.conv_base(x, edge_index)
        m = self.conv_mu(h, edge_index).relu()
        s = self.conv_logstd(h, edge_index).relu()
        return m, s

class BrainGAE_model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_site, ratio, alpha, temperature=1, site_adapt=False):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ep = VGAE(EdgePredictor(in_channels))
        self.classifier = GraphClassifier(in_channels, out_channels, num_site, ratio, site_adapt)
        self.site_adapt = site_adapt
        # if site_adapt:
        #     self.siteclassifier = site_classifier(2*dim, num_site)
        # else:
        #     self.siteclassifier = None

    def edgesampling(self, adj_logits, adj_org):
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = self.alpha*edge_probs + (1-self.alpha)*adj_org
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        adj_sampled = adj_sampled.triu(0) + torch.transpose(adj_sampled.triu(1), 1, 2)
        return adj_sampled, dense_to_sparse(adj_sampled)[0], dense_to_sparse(adj_sampled)[1]

    def forward(self, data, edge_index, edge_attr):
        x, batch = data.x, data.batch
        
        # Edge predict
        z = self.ep.encode(x, edge_index)
        z = z.view(-1, 200, 200)

        adj_logits = torch.matmul(z, torch.transpose(z, 1, 2))
        adj_logits = adj_logits.sigmoid()

        edge_dense_org = self.to_dense_(edge_index, batch, edge_attr)
        edge_dense_sampled, edge_index, edge_attr = self.edgesampling(adj_logits, edge_dense_org)
        if not self.site_adapt: 
            xy, attn1_sig, attn2_sig = self.classifier(data, edge_index,edge_attr)
            return xy, attn1_sig, attn2_sig, edge_dense_org, edge_dense_sampled, edge_index
        else:
            xy, xs, attn1_sig, attn2_sig = self.classifier(data, edge_index, edge_attr)
            return xy, xs, attn1_sig, attn2_sig, edge_dense_org, edge_dense_sampled, edge_index

    def renormalization(self):
        pass

    @staticmethod
    def to_dense_(edge_index, batch, edge_attr):
        edge_dense = to_dense_adj(edge_index, batch, edge_attr).squeeze()
        edge_dense = (edge_dense > torch.zeros_like(edge_dense)).type_as(edge_dense)
        return edge_dense

class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_site, ratio, site_adapt=True):
        super().__init__()
        dim = 64
        self.conv1 = GCNConv(in_channels, dim, cached=True, normalize=False)
        self.pool1 = TopKPooling(dim, ratio=ratio)
        self.conv2 = GCNConv(dim, dim, cached=True, normalize=False)
        self.pool2 = TopKPooling(dim, ratio=ratio)
        self.fc1 = nn.Linear(2*dim, 4*dim) #2dim, 4dim
        self.bn1 = nn.BatchNorm1d(4*dim)
        self.fc2 = nn.Linear(4*dim, 8*dim)
        self.bn2 = nn.BatchNorm1d(8*dim)
        self.fc3 = nn.Linear(8*dim, out_channels)

        if site_adapt:
            self.siteclassifier = site_classifier(2*dim, num_site)
        else:
            self.siteclassifier = None

    def forward(self, data, edge_index, edge_attr):
        x, batch = data.x, data.batch
        x = F.relu(self.conv1(x, edge_index, edge_weight=None))
        x, edge_index, edge_attr, batch, _, attn1 =self.pool1(x, edge_index, edge_attr, batch) #(160, 64))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) #(1,2dim)

        x = F.relu(self.conv2(x, edge_index, edge_weight=None))
        x, edge_index, edge_attr, batch, _, attn2 = self.pool2(x, edge_index, edge_attr, batch) #(160, 64))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) #(1, 2dim)

        x = x1 + x2
        xy = self.bn1(F.relu(self.fc1(x)))
        xy = F.dropout(xy, p=0.6, training=self.training)
        xy = self.bn2(F.relu(self.fc2(xy)))
        xy = F.log_softmax(self.fc3(xy), dim=-1)

        if self.siteclassifier is not None: 
            xs = self.siteclassifier(x)
            return xy, xs, torch.sigmoid(attn1.view(x.size(0),-1)), torch.sigmoid(attn2.view(x.size(0),-1))
        return xy, torch.sigmoid(attn1.view(x.size(0),-1)), torch.sigmoid(attn2.view(x.size(0),-1))
