#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/03/19

author: fenia
"""

import torch
from torch import nn
torch.set_printoptions(profile="full")


class WalkLayer(nn.Module):
    def __init__(self, input_size, iters=0, beta=0.9, device=-1):
        """
        Walk Layer --> Walk on the edges of a predefined graph
        Args:
            input_size (int): input dimensionality
            iters (int): number of iterations --> 2^{iters} = walks-length
            beta (float): weight shorter/longer walks
        Return:
            pairs (Tensor): final pair representations
                            size (batch * nodes * nodes, features)
        """
        super(WalkLayer, self).__init__()

        self.W = nn.Parameter(nn.init.normal_(torch.empty(input_size, input_size)), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.beta = beta
        self.iters = iters
        self.device = device

    @staticmethod
    def init_graph(graph, adj):
        """
        Initialize graph with 0 connections based on the adjacency matrix
        """
        graph = torch.where(adj.unsqueeze(-1), graph, torch.zeros_like(graph))
        return graph

    @staticmethod
    def mask_invalid_paths(graph, mask3d):
        """
        Mask invalid paths
            *(any node) -> A           -> A
            A           -> A           -> *(any node)
            A           -> *(any node) -> A

        Additionally mask paths that involve padded entities as intermediate nodes
        -inf so that sigmoid returns 0
        """
        items = range(graph.size(1))
        graph[:, :, items, items] = float('-inf')  # *->A->A
        graph[:, items, items] = float('-inf')     # A->A->*
        graph[:, items, :, items] = float('-inf')  # A->*->A (self-connection)

        graph = torch.where(mask3d.unsqueeze(-1), graph, torch.as_tensor([float('-inf')]).to(graph.device))  # padded
        graph = torch.where(torch.eq(graph, 0.0).all(dim=4, keepdim=True),
                            torch.as_tensor([float('-inf')]).to(graph.device),
                            graph)  # remaining (make sure the whole representation is zero)
        return graph

    def generate(self, old_graph):
        """
        Walk-generation: Combine consecutive edges.
        Returns: previous graph,
                 extended graph with intermediate node connections (dim=2)
        """
        graph = torch.matmul(old_graph, self.W[None, None])  # (B, I, I, D)
        graph = torch.einsum('bijk, bjmk -> bijmk', graph, old_graph)  # (B, I, I, I, D) -> dim=2 intermediate node
        return old_graph, graph

    def aggregate(self, old_graph, new_graph):
        """
        Walk-aggregation: Combine multiple paths via intermediate nodes.
        """
        # if the new representation is zero (i.e. impossible path), keep the original one --> [beta = 1]
        beta_mat = torch.where(torch.isinf(new_graph).all(dim=2),
                               torch.ones_like(old_graph),
                               torch.full_like(old_graph, self.beta))

        new_graph = self.sigmoid(new_graph)
        new_graph = torch.sum(new_graph, dim=2)  # non-linearity & sum pooling
        new_graph = torch.lerp(new_graph, old_graph, weight=beta_mat)
        return new_graph

    def forward(self, graph, adj_=None, mask_=None):
        graph = self.init_graph(graph, adj_)

        for _ in range(0, self.iters):
            old_graph, graph = self.generate(graph)
            graph = self.mask_invalid_paths(graph, mask_)
            graph = self.aggregate(old_graph, graph)
        return graph
