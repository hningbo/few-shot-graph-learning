"""
@Filename       : gnn.py
@Create Time    : 2022/5/15 17:10
@Author         : Rylynn
@Description    : 

"""
import torch as th
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv


class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats=config['input_size'], out_feats=config['hidden_size'])
        self.conv2 = GraphConv(in_feats=config['hidden_size'], out_feats=config['output_size'])

        # self.relu =
    
    def forward(self, g, feat):
        out = self.conv1(g, feat)
        out = self.conv2(g, out)


if __name__ == '__main__':
    ...