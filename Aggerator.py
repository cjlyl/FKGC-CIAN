"""
The part of codes is from FAAN ( https://github.com/JiaweiSheng/FAAN)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable


class SoftSelectAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SoftSelectAttention, self).__init__()


    def forward(self, support, query):
        """
        :param support: [few_shot, dim]
        :param query: [batch_size, dim]
        :return:
        """
        query_ = query.unsqueeze(1).expand(query.size(0), support.size(0), query.size(1)).contiguous()  # [b, few, dim]
        support_ = support.unsqueeze(0).expand_as(query_).contiguous()  # [b, few, dim]

        scalar = support.size(1) ** -0.5  # dim ** -0.5
        score = torch.sum(query_ * support_, dim=2) * scalar
        att = torch.softmax(score, dim=1)  #(128,5)

        center = torch.mm(att, support)
        return center

class SoftSelectPrototype(nn.Module):
    def __init__(self, r_dim):
        super(SoftSelectPrototype, self).__init__()
        self.Attention = SoftSelectAttention(hidden_size=r_dim)

    def forward(self, support, query):
        center = self.Attention(support, query)
        return center