import timeit
import os
import torch
import numpy as np
from scipy.io import loadmat
from torch_geometric.data import Data
import networkx
from networkx.convert_matrix import from_numpy_matrix
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import deepdish as dd
import torch_geometric.transforms as T

EPS = 1E-10
EDGE = 20

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])     #nodes
    print(node_slice)
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])     #edges
    print(edge_slice)
    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice
    if data.site is not None:
        if data.site.size(0) == batch.size(0):
            slices['site'] = node_slice
        else:
            slices['site'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None


def read_data(data_dir, augment = True):
    onlyfiles = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    site_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    node_list = []
    node_list.append(EDGE)
    if augment:
        node_list.append(20)
        node_list.append(20)
        
    start = timeit.default_timer()
    with multiprocessing.Pool(os.cpu_count()-1) as pool:
        for i, node in enumerate(node_list):
            # print(node)
            func = partial(read_sigle_data, data_dir, node)
            res = pool.map(func, onlyfiles)
            for j in range(len(res)):
                edge_att_list.append(res[j][0])
                edge_index_list.append(res[j][1]+(j+i*len(res))*res[j][4])
                att_list.append(res[j][2])
                y_list.append(res[j][3])
                site_list.append(res[j][5])
                batch.append([j+i*len(res)]*res[j][4])
                # print(batch)
                pseudo.append(np.diag(np.ones(res[j][4])))
            edge_att_arr = np.concatenate(edge_att_list)
            edge_index_arr = np.concatenate(edge_index_list, axis=1)
            att_arr = np.concatenate(att_list, axis=0)
            pseudo_arr = np.concatenate(pseudo, axis=0)
            y_arr = np.stack(y_list)
            site_arr = np.stack(site_list)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    edge_att_torch = torch.FloatTensor(edge_att_arr.reshape(len(edge_att_arr), 1))
    att_torch = torch.FloatTensor(att_arr)
    y_torch = torch.LongTensor(y_arr)  # classification
    site_torch = torch.LongTensor(site_arr)
    batch_torch = torch.LongTensor(np.hstack(batch))
    edge_index_torch = torch.LongTensor(edge_index_arr)
    # print(edge_index_torch.shape)
    pseudo_torch = torch.FloatTensor(pseudo_arr)
    # print(batch_torch.shape)
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, 
                edge_attr=edge_att_torch, pos = pseudo_torch, site=site_torch )
    data, slices = split(data, batch_torch)
    return data, slices


def read_sigle_data(data_dir, numnode, filename, transform=True):
    temp = dd.io.load(os.path.join(data_dir, filename))

    # read edge attribute, node features, and labels
    pcorr = np.abs(temp['pcorr'])# edge attributes

    # get top_k matridxx
    if not transform:
        pcorr = get_top_k_matrix(pcorr, numnode)
        # print(pcorr)
        # print(np.count_nonzero(pcorr))
    att = temp['corr'][()] # node features
    label = temp['label'][()] # labels
    site = temp['site'] # site onehot encode

    num_nodes = pcorr.shape[0]
    G = from_numpy_matrix(pcorr)
    A = networkx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()

    edge_att = np.zeros(len(adj.row))
    # print(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]
    edge_index = np.stack([adj.row, adj.col])
    edge_index = torch.LongTensor(edge_index)
    edge_att = torch.FloatTensor(edge_att)

    # if remove_self_loop:
    edge_index, edge_att = remove_self_loops(edge_index, edge_att)

    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)
    # print(edge_index.shape)
    data = Data(x=torch.FloatTensor(att), 
                edge_index=edge_index, 
                y=torch.LongTensor(label), 
                edge_attr=edge_att,
                site=torch.LongTensor(site))
    
    if transform:
        gdc = T.GDC(self_loop_weight=None, 
                    normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=0.05),
                    sparsification_kwargs=dict(method='topk', k=numnode, dim=0), 
                    exact=True,
                    )
        data = gdc(data)
        return data.edge_attr.data.numpy(), data.edge_index.data.numpy(), data.x.data.numpy(),data.y.data.item(),num_nodes,site
    else:
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes,site

def get_top_k_matrix(A: np.ndarray, k: int):
    adj = np.ones_like(A)
    num_nodes = A.shape[1]
    row_index = np.arange(num_nodes)
    adj[A.argsort(axis=0)[:num_nodes-k],row_index] = EPS
    return adj

if __name__ == "__main__":
    data_dir = '../data/ABIDE_pcp/cpac/filt_global/raw'
    filename = '51478.h5'
    read_sigle_data(data_dir, 20, filename)
    # read_data(data_dir, False)    