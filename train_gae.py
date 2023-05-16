from models.braingae import BrainGAE
import argparse
import glob
import os
import time

from imports.ABIDEDataset import ABIDEDataset
from utils.process import train_val_test_split
from torch_geometric.loader import DataLoader


# Some setup constants
PRJ_PATH = os.getcwd()
NAME = 'ABIDE'
PATH = os.path.join(PRJ_PATH, 'data/ABIDE_pcp/cpac/filt_global/')
NUMSITE = 20

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nheads', type=int, default=3, help='Number of attention heads.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--lambda0', type=float, default=1, help='Class loss factor')
    parser.add_argument('--lambda1', type=float, default=0.2, help='Site loss factor')
    parser.add_argument('--lambda2', type=float, default=0.1, help='Attention loss factor')
    parser.add_argument('--lambda3', type=float, default=0.5, help='Edge loss factor')
    parser.add_argument('--alpha', type=float, default=0.5, help='Edge drop factor')
    parser.add_argument('--beta', type=float, default=0.5, help='Hmm factor')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--patience', type=int, default=50, help='Patience')
    parser.add_argument('--batch', type=int, default=100, help='Batch size')
    parser.add_argument('--fold', type=int, default=0, help='K-fold')
    parser.add_argument('--seed', type=int, default=123, help='Seed for RNG')
    parser.add_argument('--ratio', type=float, default=0.5, help='Pooling ratio')
    parser.add_argument('--aggr', type=str, default='add', help='Node aggregation')
    parser.add_argument('--gnn', type=str, default='gcn', help='GNN Layer')
    parser.add_argument('--nosite', action='store_true', default=False, help='Using site adaptation')
    parser.add_argument('--pretrain-cl', action='store_true', default=False, help='Pretrain classifier net')
    parser.add_argument('--pretrain-ep', action='store_true', default=False, help='Pretrain ep net')
    parser.add_argument('--ep', action='store_true', default=False, help='ep net')
    parser.add_argument('--mix', action='store_true', default=False, help='Mix final adj')
    args = parser.parse_args()

    # Load data
    dataset = ABIDEDataset(PATH, NAME, False)

    dataset.data.y = dataset.data.y.squeeze()
    if not args.nosite:
        dataset.data.site = dataset.data.site.squeeze()

    idx_train, idx_val, idx_test = train_val_test_split(kfold=5, fold=args.fold, seed=132)

    train_dataset = dataset[idx_train]
    val_dataset = dataset[idx_val]
    test_dataset = dataset[idx_test]
    batch_val = args.batch
    if args.mix:
        batch_val = 1
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_val, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_val, shuffle=False)

    model = BrainGAE(in_channels=dataset.num_features,
                     out_channels=dataset.num_classes,
                     num_site=NUMSITE,
                     ratio=args.ratio,
                     alpha=args.alpha,
                     n_epochs=args.epochs,
                     lr=args.lr,
                     weight_decay=args.weight_decay,
                     temperature=1,
                     site_adapt=not args.nosite,
                     augment=False,
                     train_loader=train_loader,
                     val_loader=val_loader,
                     test_loader=test_loader,
                     threshold=0.05,
                     l0=args.lambda0,
                     l1=args.lambda1,
                     l2=args.lambda2,
                     l3=args.lambda3,
                     pretrain_cl=args.pretrain_cl,
                     pretrain_ep=args.pretrain_ep,
                     beta=args.beta,
                    #  sampling='bernouli',
                     sampling='addremove',
                     mix=args.mix,
                     aggr=args.aggr,
                     gnn=args.gnn,
                     edgepredictor=args.ep
                     )
    model.fit(args.patience)

