import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable
from AttentionModule import Attention_Module
from Aggerator import *


class CIAN(nn.Module):
    def __init__(self, args, num_symbols, embedding_size, embed, use_pretrain, finetune):
        super(CIAN, self).__init__()

        self.args = args
        self.entity_encoder = Attention_Module(self.args, embed=embed, num_symbols = num_symbols,
                                                embedding_size = embedding_size,
                                                use_pretrain=use_pretrain, finetune=finetune)


        self.Aggerator = SoftSelectPrototype(self.args)


    def forward(self, support, support_meta,
                query, query_meta,
                false = None, false_meta=None, is_train=True):


        if is_train:

            support_rep = self.entity_encoder(support, support_meta)
            query_rep = self.entity_encoder(query, query_meta)
            false_rep = self.entity_encoder(false, false_meta)
            support_query = self.Aggerator(support_rep, query_rep) #(1, emb_dim) (128,100)
            support_false = self.Aggerator(support_rep, false_rep) #(1, emb_dim)



            positive_score = torch.sum(query_rep * support_query, dim=1) #size:([batch_size])
            negative_score = torch.sum(false_rep * support_false, dim=1)


        else:
            support_rep = self.entity_encoder(support, support_meta)
            query_rep = self.entity_encoder(query, query_meta)
            support_query = self.Aggerator(support_rep, query_rep) #(1, emb_dim) (128,100)

            positive_score = torch.sum(query_rep * support_query, dim=1)
            negative_score = None
            # sim = None


        return positive_score, negative_score