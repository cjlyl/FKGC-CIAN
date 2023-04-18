

import numpy as np
import torch
import torch.nn as nn
import logging
import torch.nn.init as init
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(Attention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size, bias=False)

        self.att_dropout = nn.Dropout(0.2)
        self.Bilinear_att = nn.Linear(self.att_size, self.att_size, bias=False)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        """
        q (target_rel):  (few/b, 1, dim)
        k (nbr_rel):    (few/b, max, dim)
        v (nbr_ent):    (few/b, max, dim)
        mask:   (few/b, max)
        output:
        """
        q = q.unsqueeze(1)
        orig_q_size = q.size()
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.att_size)   #(few/b, 1, num_heads, att_size)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.att_size)   #(few/b, max, num_heads, att_size)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.att_size)   #(few/b, max, num_heads, att_size)


        q = q.transpose(1, 2)  #(few/b, num_heads, 1, att_size)
        k = k.transpose(1, 2).transpose(2, 3)  #(few/b, num_heads, att_size, max)
        v = v.transpose(1, 2)  #(few/b, num_heads, max, att_size)

        x = torch.matmul(self.Bilinear_att(q), k)

        x = torch.softmax(x, dim=3)   # [few/b, num_heads, 1, max]

        x = self.att_dropout(x)     # [few/b, num_heads, 1, max]
        x = x.matmul(v)    #(few/b, num_heads, 1, att_size)

        x = x.transpose(1, 2).contiguous()  # (few/b, 1, num_heads, att_size)

        x = x.view(batch_size, -1, self.num_heads * self.att_size).squeeze(1) #(few/b, dim)
        x = self.output_layer(x)  #(few/b, dim)

        return x



class Attention_Module(nn.Module):
    def __init__(self, args, embed, num_symbols, embedding_size, use_pretrain=True, finetune=True, dropout_rate=0.3):
        super(Attention_Module, self).__init__()

        self.agrs = args
        self.embedding_size = embedding_size
        self.pad_idx = num_symbols
        self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).to(args.device)

        self.symbol_emb = nn.Embedding(num_symbols+1, self.embedding_size, padding_idx=self.pad_idx)
        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_MLP = nn.Dropout(0.2)

        self.LeakyRelu = nn.LeakyReLU()
        self.task_aware_attention_module = Attention(hidden_size=self.embedding_size, num_heads=1) #
        self.entity_pair_attention_module = Attention(hidden_size=self.embedding_size, num_heads=1) #

        self.layer_norm = nn.LayerNorm(self.embedding_size)


        self.gate_w = nn.Linear(2*self.embedding_size, self.embedding_size) #

        self.Linear_tail = nn.Linear(self.embedding_size, self.embedding_size, bias=False) #
        self.Linear_head = nn.Linear(self.embedding_size, self.embedding_size, bias=False)


        self.rel_w = nn.Bilinear(self.embedding_size, self.embedding_size, 2* self.embedding_size, bias=False) #2*

        self.MLPW1 = nn.Linear(2 * self.embedding_size, 2 * self.embedding_size)

        self.MLPW2 = nn.Linear(2 * self.embedding_size, 2 * self.embedding_size, bias=False)

        init.xavier_normal_(self.MLPW1.weight)
        init.xavier_normal_(self.MLPW2.weight)
        init.xavier_normal_(self.gate_w.weight)

        self.layer_norm1 = nn.LayerNorm(2 * self.embedding_size)

    def bi_MLP(self, input):
        output = torch.relu(self.MLPW1(input))
        output = self.MLPW2(output)
        output = self.layer_norm1(output + input)
        return output


    def SPD_attention(self, entity_left, entity_right, rel_emb_forward, rel_emb_backward,
                      rel_embeds_left, rel_embeds_right, ent_embeds_left, ent_embeds_right):
        """
        ent： (few/b, dim)
        nn:  (few/b, max, 2*dim)
        output:  ()
        """
        V_head = torch.relu(self.gate_w(torch.cat([rel_embeds_left, ent_embeds_left], dim=-1)))
        V_tail = torch.relu(self.gate_w(torch.cat([rel_embeds_right, ent_embeds_right], dim=-1)))

        # learning h's task-aware representation
        head_nn_rel_aware = self.task_aware_attention_module(q=rel_emb_forward, k=rel_embeds_left, v=V_head)  # (few/b, dim)
        head_nn_rel_aware = torch.relu(self.Linear_tail(head_nn_rel_aware) + self.Linear_head(entity_left))
        enhanced_head_ = self.layer_norm(head_nn_rel_aware + entity_left)

        # learning t's task-aware representation
        tail_nn_rel_aware = self.task_aware_attention_module(q=rel_emb_backward, k=rel_embeds_right, v=V_tail)  # (few/b, dim)
        tail_nn_rel_aware = torch.relu(self.Linear_tail(tail_nn_rel_aware) + self.Linear_head(entity_right))
        enhanced_tail_ = self.layer_norm(tail_nn_rel_aware + entity_right)


        # learning h's entity-pair-aware representation
        head_nn_ent_aware = self.entity_pair_attention_module(q=enhanced_tail_, k=ent_embeds_left, v=V_head)
        head_nn_ent_aware = torch.relu(self.Linear_tail(head_nn_ent_aware) + self.Linear_head(entity_left))
        enhanced_head = self.layer_norm(head_nn_ent_aware + entity_left)


        # learning t's entity-pair-aware representation
        tail_nn_ent_aware = self.entity_pair_attention_module(q=enhanced_head_, k=ent_embeds_right, v=V_tail)
        tail_nn_ent_aware = torch.relu(self.Linear_tail(tail_nn_ent_aware) + self.Linear_head(entity_right))
        enhanced_tail = self.layer_norm(tail_nn_ent_aware + entity_right)

        # computing entity-pair representation
        enhanced_pair = torch.cat([enhanced_head, enhanced_tail], dim=-1)
        ent_pair_rep = self.bi_MLP(enhanced_pair)
        return ent_pair_rep


    def forward(self, entity_pairs, entity_meta):

        entity = self.dropout(self.symbol_emb(entity_pairs))  # (few/b, 2, dim)
        entity_left, entity_right = torch.split(entity, 1, dim=1)  # (few/b, 1, dim)
        entity_left = entity_left.squeeze(1)    # (few/b, dim)
        entity_right = entity_right.squeeze(1)   # (few/b, dim)


        entity_left_connections, entity_left_degrees, entity_right_connections, entity_right_degrees = entity_meta

        relations_left = entity_left_connections[:, :, 0].squeeze(-1)
        entities_left = entity_left_connections[:, :, 1].squeeze(-1)
        rel_embeds_left = self.dropout(self.symbol_emb(relations_left))  # (few/b, max, dim)
        ent_embeds_left = self.dropout(self.symbol_emb(entities_left))   # (few/b, max, dim)

        relations_right = entity_right_connections[:, :, 0].squeeze(-1)
        entities_right = entity_right_connections[:, :, 1].squeeze(-1)
        rel_embeds_right = self.dropout(self.symbol_emb(relations_right))  # (few/b, max, dim)
        ent_embeds_right = self.dropout(self.symbol_emb(entities_right)) # (few/b, max, dim)

        rel_emb = self.rel_w(entity_left, entity_right)
        rel_emb_forward, rel_emb_backward = torch.split(rel_emb, entity_left.size(-1), dim=-1)  # (few/b, dim)

        ent_pair_rep = self.SPD_attention(entity_left, entity_right, rel_emb_forward, rel_emb_backward,
                                          rel_embeds_left, rel_embeds_right, ent_embeds_left, ent_embeds_right)


        return ent_pair_rep

