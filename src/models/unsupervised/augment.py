"""
@Filename       : augment.py
@Create Time    : 2022/5/17 16:24
@Author         : Rylynn
@Description    : 

"""
import copy

import torch as th
import dgl
import random
import numpy as np
import scipy.sparse as sp

class Augmentor():
    def __init__(self):
        ...

    def delete_row_col(self, input_matrix, drop_list, only_row=False):
        remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
        out = input_matrix[remain_list, :]
        if only_row:
            return out
        out = out[:, remain_list]

        return out

    def attribute_mask(self, g, feat, drop_percent):
        node_num = feat.shape[1]
        mask_num = int(node_num * drop_percent)
        node_idx = [i for i in range(node_num)]
        mask_idx = random.sample(node_idx, mask_num)
        aug_feature = copy.deepcopy(feat)
        zeros = th.zeros_like(aug_feature[0][0])
        for j in mask_idx:
            aug_feature[0][j] = zeros
        return g, aug_feature

    def node_dropping(self, g5, feat, drop_percent):
        node_num = g.number_of_nodes()
        drop_num = int(node_num * drop_percent)  # number of drop nodes
        all_node_list = [i for i in range(node_num)]
        drop_node_list = sorted(random.sample(all_node_list, drop_num))

        aug_g = copy.deepcopy(g)
        aug_g.remove_nodes(drop_node_list)

        aug_feat = self.delete_row_col(feat, drop_node_list, only_row=True)

        return aug_g, aug_feat

    def edge_perturbation(self, g, drop_percent):
        percent = drop_percent / 2
        aug_g = copy.deepcopy(g)

        row_idx, col_idx = input_adj.nonzero()

        index_list = []
        for i in range(len(row_idx)):
            index_list.append((row_idx[i], col_idx[i]))

        single_index_list = []
        for i in list(index_list):
            single_index_list.append(i)
            index_list.remove((i[1], i[0]))

        edge_num = int(len(row_idx) / 2)  # 9228 / 2
        add_drop_num = int(edge_num * percent / 2)
        aug_adj = copy.deepcopy(input_adj.todense().tolist())

        edge_idx = [i for i in range(edge_num)]
        drop_idx = random.sample(edge_idx, add_drop_num)

        for i in drop_idx:
            aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
            aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0

        '''
        above finish drop edges
        '''
        node_num = input_adj.shape[0]
        l = [(i, j) for i in range(node_num) for j in range(i)]
        add_list = random.sample(l, add_drop_num)

        for i in add_list:
            aug_adj[i[0]][i[1]] = 1
            aug_adj[i[1]][i[0]] = 1

        aug_adj = np.matrix(aug_adj)
        aug_adj = sp.csr_matrix(aug_adj)
        return aug_adj

    def sampling_random_walk(self, g, feat):
        ...

    def sampling_unified(self, g, feat):
        ...

    def sampling_ego_network(self, g, feat):
        ...



if __name__ == '__main__':
    augmentor = Augmentor()
    g: dgl.DGLGraph = dgl.graph(data=([1, 2, 3],[4, 5, 6]))

    print(augmentor.)
