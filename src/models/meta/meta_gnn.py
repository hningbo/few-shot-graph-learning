"""
@Filename       : meta_gnn.py
@Create Time    : 2022/5/15 17:10
@Author         : Rylynn
@Description    : 

"""

import torch as th
import torch.nn as nn

class MetaGNN(nn.Module):
    def __init__(self, model):
        super(MetaGNN, self).__init__()
        self.model = model
        self.cross_ent = nn.CrossEntropyLoss()

    def forward(self, sup_x, sup_y, qry_x, qry_y):
        sup_pred_x = self.model(sup_x)
        sup_loss = self.classification_loss(sup_x, sup_y)

        qry_pred_y = self.model(qry_x)
        qry_loss = self.classification_loss(qry_x, qry_y)
        ...

    def classification_loss(self, x, y):
        return self.cross_ent(x, y)

    def loss(self):
        ...