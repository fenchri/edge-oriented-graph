#!/usr/bin/env python3
"""
Created on 29/03/19

author: fenia
"""

import torch
from torch import nn, torch
import math


class Dot_Attention(nn.Module):
    """
    Adaptation from "Attention is all you need".
    Here the query is the target pair and the keys/values are the words of the sentence.
    The dimensionality of the queries and the values should be the same.
    """
    def __init__(self, input_size, device=-1, scale=False):

        super(Dot_Attention, self).__init__()

        self.softmax = nn.Softmax(dim=2)
        self.scale = scale
        if scale:
            self.sc = 1.0 / math.sqrt(input_size)
        self.device = device

    def create_mask(self, alpha, size_, lengths, idx_):
        """ Put 1 in valid tokens """
        mention_sents = torch.index_select(lengths, 0, idx_[:, 4])

        # mask padded words (longer that sentence length)
        tempa = torch.arange(size_).unsqueeze(0).repeat(alpha.shape[0], 1).to(self.device)
        mask_a = torch.ge(tempa, mention_sents[:, None])

        # mask tokens that are used as queries
        tempb = torch.arange(lengths.size(0)).unsqueeze(0).repeat(alpha.shape[0], 1).to(self.device)  # m x sents
        sents = torch.where(torch.lt(tempb, idx_[:, 4].unsqueeze(1)),
                            lengths.unsqueeze(0).repeat(alpha.shape[0], 1),
                            torch.zeros_like(lengths.unsqueeze(0).repeat(alpha.shape[0], 1)))

        total_l = torch.cumsum(sents, dim=1)[:, -1]
        mask_b = torch.ge(tempa, (idx_[:, 2] - total_l)[:, None]) & torch.lt(tempa, (idx_[:, 3] - total_l)[:, None])

        mask = ~(mask_a | mask_b)
        del tempa, tempb, total_l
        return mask

    def forward(self, queries, values, idx, lengths):
        """
        a = softmax( q * H^T )
        v = a * H
        """
        alpha = torch.matmul(queries.unsqueeze(1), values.transpose(1, 2))

        if self.scale:
            alpha = alpha * self.sc

        mask_ = self.create_mask(alpha, values.size(1), lengths, idx)
        alpha = torch.where(mask_.unsqueeze(1),
                            alpha,
                            torch.as_tensor([float('-inf')]).to(self.device))
        alpha = self.softmax(alpha)
        alpha = torch.squeeze(alpha, 1)
        return alpha
