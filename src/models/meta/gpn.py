"""
@Filename       : gpn.py
@Create Time    : 2022/5/15 22:29
@Author         : Rylynn
@Description    : 

"""

import torch as th
import torch.nn as nn


class GPN(nn.Module):
    def __init__(self):
        super(GPN, self).__init__()

    def forward(self, input):
        ...