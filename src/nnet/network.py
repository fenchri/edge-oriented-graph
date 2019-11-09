#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25-Feb-2019

author: fenia
"""

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from nnet.init_net import BaseNet


class EOG(BaseNet):
    def input_layer(self, words_):
        """
        Word Embedding Layer
        """
        word_vec = self.word_embed(words_)
        return word_vec

    def encoding_layer(self, word_vec, word_sec):
        """
        Encoder Layer -> Encode sequences using BiLSTM.
        """
        ys = self.encoder(torch.split(word_vec, word_sec.tolist(), dim=0), word_sec)
        return ys

    def graph_layer(self, encoded_seq, info, word_sec, section, positions):
        """
        Graph Layer -> Construct a document-level graph
        The graph edges hold representations for the connections between the nodes.
        Args:
            encoded_seq: Encoded sequence, shape (sentences, words, dimension)
            info:        (Tensor, 5 columns) entity_id, entity_type, start_wid, end_wid, sentence_id
            word_sec:    (Tensor) number of words per sentence
            section:     (Tensor <B, 3>) #entities/#mentions/#sentences per batch
            positions:   distances between nodes (only M-M and S-S)

        Returns: (Tensor) graph, (Tensor) tensor_mapping, (Tensors) indices, (Tensor) node information
        """
        # SENTENCE NODES
        sentences = torch.mean(encoded_seq, dim=1)  # sentence nodes (avg of sentence words)

        # MENTION & ENTITY NODES
        temp_ = torch.arange(word_sec.max()).unsqueeze(0).repeat(sentences.size(0), 1).to(self.device)
        remove_pad = (temp_ < word_sec.unsqueeze(1))

        mentions = self.merge_tokens(info, encoded_seq, remove_pad)  # mention nodes
        entities = self.merge_mentions(info, mentions)               # entity nodes

        # all nodes in order: entities - mentions - sentences
        nodes = torch.cat((entities, mentions, sentences), dim=0)  # e + m + s (all)
        nodes_info = self.node_info(section, info)                 # info/node: node type | semantic type | sentence ID

        if self.types:  # + node types
            nodes = torch.cat((nodes, self.type_embed(nodes_info[:, 0])), dim=1)

        # re-order nodes per document (batch)
        nodes = self.rearrange_nodes(nodes, section)
        nodes = self.split_n_pad(nodes, section, pad=0)

        nodes_info = self.rearrange_nodes(nodes_info, section)
        nodes_info = self.split_n_pad(nodes_info, section, pad=-1)

        # create initial edges (concat node representations)
        r_idx, c_idx = torch.meshgrid(torch.arange(nodes.size(1)).to(self.device),
                                      torch.arange(nodes.size(1)).to(self.device))
        graph = torch.cat((nodes[:, r_idx], nodes[:, c_idx]), dim=3)
        r_id, c_id = nodes_info[..., 0][:, r_idx], nodes_info[..., 0][:, c_idx]  # node type indicators

        # pair masks
        pid = self.pair_ids(r_id, c_id)

        # Linear reduction layers
        reduced_graph = torch.where(pid['MS'].unsqueeze(-1), self.reduce['MS'](graph),
                                    torch.zeros(graph.size()[:-1] + (self.out_dim,)).to(self.device))
        reduced_graph = torch.where(pid['ME'].unsqueeze(-1), self.reduce['ME'](graph), reduced_graph)
        reduced_graph = torch.where(pid['ES'].unsqueeze(-1), self.reduce['ES'](graph), reduced_graph)

        if self.dist:
            dist_vec = self.dist_embed(positions)   # distances
            reduced_graph = torch.where(pid['SS'].unsqueeze(-1),
                                        self.reduce['SS'](torch.cat((graph, dist_vec), dim=3)), reduced_graph)
        else:
            reduced_graph = torch.where(pid['SS'].unsqueeze(-1), self.reduce['SS'](graph), reduced_graph)

        if self.context and self.dist:
            m_cntx = self.attention(mentions, encoded_seq[info[:, 4]], info, word_sec)
            m_cntx = self.prepare_mention_context(m_cntx, section, r_idx, c_idx,
                                                  encoded_seq[info[:, 4]], pid, nodes_info)

            reduced_graph = torch.where(pid['MM'].unsqueeze(-1),
                                        self.reduce['MM'](torch.cat((graph, dist_vec, m_cntx), dim=3)), reduced_graph)

        elif self.context:
            m_cntx = self.attention(mentions, encoded_seq[info[:, 4]], info, word_sec)
            m_cntx = self.prepare_mention_context(m_cntx, section, r_idx, c_idx,
                                                  encoded_seq[info[:, 4]], pid, nodes_info)

            reduced_graph = torch.where(pid['MM'].unsqueeze(-1),
                                        self.reduce['MM'](torch.cat((graph, m_cntx), dim=3)), reduced_graph)

        elif self.dist:
            reduced_graph = torch.where(pid['MM'].unsqueeze(-1),
                                        self.reduce['MM'](torch.cat((graph, dist_vec), dim=3)), reduced_graph)

        else:
            reduced_graph = torch.where(pid['MM'].unsqueeze(-1), self.reduce['MM'](graph), reduced_graph)

        if self.ee:
            reduced_graph = torch.where(pid['EE'].unsqueeze(-1), self.reduce['EE'](graph), reduced_graph)

        mask = self.get_nodes_mask(section.sum(dim=1))
        return reduced_graph, (r_idx, c_idx), nodes_info, mask

    def prepare_mention_context(self, m_cntx, section, r_idx, c_idx, s_seq, pid, nodes_info):
        """
        Estimate attention scores for each pair
        (a1 + a2)/2 * sentence_words
        """
        # "fake" mention weight nodes
        m_cntx = torch.cat((torch.zeros(section.sum(dim=0)[0], m_cntx.size(1)).to(self.device),
                            m_cntx,
                            torch.zeros(section.sum(dim=0)[2], m_cntx.size(1)).to(self.device)), dim=0)
        m_cntx = self.rearrange_nodes(m_cntx, section)
        m_cntx = self.split_n_pad(m_cntx, section, pad=0)
        m_cntx = torch.div(m_cntx[:, r_idx] + m_cntx[:, c_idx], 2)

        # mask non-MM pairs
        # mask invalid weights (i.e. M-M not in the same sentence)
        mask_ = torch.eq(nodes_info[..., 2][:, r_idx], nodes_info[..., 2][:, c_idx]) & pid['MM']
        m_cntx = torch.where(mask_.unsqueeze(-1), m_cntx, torch.zeros_like(m_cntx))

        # "fake" mention sentences nodes
        sents = torch.cat((torch.zeros(section.sum(dim=0)[0], m_cntx.size(3), s_seq.size(2)).to(self.device),
                           s_seq,
                           torch.zeros(section.sum(dim=0)[2], m_cntx.size(3), s_seq.size(2)).to(self.device)), dim=0)
        sents = self.rearrange_nodes(sents, section)
        sents = self.split_n_pad(sents, section, pad=0)
        m_cntx = torch.matmul(m_cntx, sents)
        return m_cntx

    @staticmethod
    def pair_ids(r_id, c_id):
        pids = {
            'EE': ((r_id == 0) & (c_id == 0)),
            'MM': ((r_id == 1) & (c_id == 1)),
            'SS': ((r_id == 2) & (c_id == 2)),
            'ES': (((r_id == 0) & (c_id == 2)) | ((r_id == 2) & (c_id == 0))),
            'MS': (((r_id == 1) & (c_id == 2)) | ((r_id == 2) & (c_id == 1))),
            'ME': (((r_id == 1) & (c_id == 0)) | ((r_id == 0) & (c_id == 1)))
        }
        return pids

    @staticmethod
    def rearrange_nodes(nodes, section):
        """
        Re-arrange nodes so that they are in 'Entity - Mention - Sentence' order for each document (batch)
        """
        tmp1 = section.t().contiguous().view(-1).long().to(nodes.device)
        tmp3 = torch.arange(section.numel()).view(section.size(1),
                                                  section.size(0)).t().contiguous().view(-1).long().to(nodes.device)
        tmp2 = torch.arange(section.sum()).to(nodes.device).split(tmp1.tolist())
        tmp2 = pad_sequence(tmp2, batch_first=True, padding_value=-1)[tmp3].view(-1)
        tmp2 = tmp2[(tmp2 != -1).nonzero().squeeze()]  # remove -1 (padded)

        nodes = torch.index_select(nodes, 0, tmp2)
        return nodes

    @staticmethod
    def split_n_pad(nodes, section, pad=None):
        nodes = torch.split(nodes, section.sum(dim=1).tolist())
        nodes = pad_sequence(nodes, batch_first=True, padding_value=pad)
        return nodes

    @staticmethod
    def get_nodes_mask(nodes_size):
        """
        Create mask for padded nodes
        """
        n_total = torch.arange(nodes_size.max()).to(nodes_size.device)
        idx_r, idx_c, idx_d = torch.meshgrid(n_total, n_total, n_total)

        # masks for padded elements (1 in valid, 0 in padded)
        ns = nodes_size[:, None, None, None]
        mask3d = ~(torch.ge(idx_r, ns) | torch.ge(idx_c, ns) | torch.ge(idx_d, ns))
        return mask3d

    def node_info(self, section, info):
        """
        Col 0: node type | Col 1: semantic type | Col 2: sentence id
        """
        typ = torch.repeat_interleave(torch.arange(3).to(self.device), section.sum(dim=0))  # node types (0,1,2)
        rows_ = torch.bincount(info[:, 0]).cumsum(dim=0).sub(1)
        stypes = torch.neg(torch.ones(section[:, 2].sum())).to(self.device).long()  # semantic type sentences = -1
        all_types = torch.cat((info[:, 1][rows_], info[:, 1], stypes), dim=0)
        sents_ = torch.arange(section.sum(dim=0)[2]).to(self.device)
        sent_id = torch.cat((info[:, 4][rows_], info[:, 4], sents_), dim=0)  # sent_id
        return torch.cat((typ.unsqueeze(-1), all_types.unsqueeze(-1), sent_id.unsqueeze(-1)), dim=1)

    def estimate_loss(self, pred_pairs, truth):
        """
        Softmax cross entropy loss.
        Args:
            pred_pairs (Tensor): Un-normalized pairs (# pairs, classes)
            truth (Tensor): Ground-truth labels (# pairs, id)

        Returns: (Tensor) loss, (Tensors) TP/FP/FN
        """
        mask = torch.ne(truth, -1)
        truth = truth[mask]
        pred_pairs = pred_pairs[mask]

        assert (truth != -1).all()
        loss = self.loss(pred_pairs, truth)

        predictions = F.softmax(pred_pairs, dim=1).data.argmax(dim=1)
        stats = self.count_predictions(predictions, truth)
        return loss, stats, predictions

    @staticmethod
    def merge_mentions(info, mentions):
        """
        Merge mentions into entities;
        Find which rows (mentions) have the same entity id and average them
        """
        m_ids, e_ids = torch.broadcast_tensors(info[:, 0].unsqueeze(0),
                                               torch.arange(0, max(info[:, 0]) + 1).unsqueeze(-1).to(info.device))
        index_m = torch.eq(m_ids, e_ids).type('torch.FloatTensor').to(info.device)
        entities = torch.div(torch.matmul(index_m, mentions), torch.sum(index_m, dim=1).unsqueeze(-1))  # average
        return entities

    @staticmethod
    def merge_tokens(info, enc_seq, rm_pad):
        """
        Merge tokens into mentions;
        Find which tokens belong to a mention (based on start-end ids) and average them
        """
        enc_seq = enc_seq[rm_pad]
        start, end, w_ids = torch.broadcast_tensors(info[:, 2].unsqueeze(-1),
                                                    info[:, 3].unsqueeze(-1),
                                                    torch.arange(0, enc_seq.shape[0]).unsqueeze(0).to(info.device))
        index_t = (torch.ge(w_ids, start) & torch.lt(w_ids, end)).float().to(info.device)
        mentions = torch.div(torch.matmul(index_t, enc_seq), torch.sum(index_t, dim=1).unsqueeze(-1))   # average
        return mentions

    @staticmethod
    def select_pairs(combs, nodes_info, idx):
        """
        Select (entity node) pairs for classification based on input parameter restrictions (i.e. their entity type).
        """
        combs = torch.split(combs, 2, dim=0)
        sel = torch.zeros(nodes_info.size(0), nodes_info.size(1), nodes_info.size(1)).to(nodes_info.device)

        a_ = nodes_info[..., 0][:, idx[0]]
        b_ = nodes_info[..., 0][:, idx[1]]
        c_ = nodes_info[..., 1][:, idx[0]]
        d_ = nodes_info[..., 1][:, idx[1]]
        for ca, cb in combs:
            condition1 = torch.eq(a_, 0) & torch.eq(b_, 0)      # needs to be an entity node (id=0)
            condition2 = torch.eq(c_, ca) & torch.eq(d_, cb)    # valid pair semantic types
            sel = torch.where(condition1 & condition2, torch.ones_like(sel), sel)
        return sel.nonzero().unbind(dim=1)

    def count_predictions(self, y, t):
        """
        Count number of TP, FP, FN, TN for each relation class
        """
        label_num = torch.as_tensor([self.rel_size]).long().to(self.device)
        ignore_label = torch.as_tensor([self.ignore_label]).long().to(self.device)

        mask_t = torch.eq(t, ignore_label).view(-1)          # where the ground truth needs to be ignored
        mask_p = torch.eq(y, ignore_label).view(-1)          # where the predicted needs to be ignored

        true = torch.where(mask_t, label_num, t.view(-1).long().to(self.device))  # ground truth
        pred = torch.where(mask_p, label_num, y.view(-1).long().to(self.device))  # output of NN

        tp_mask = torch.where(torch.eq(pred, true), true, label_num)
        fp_mask = torch.where(torch.ne(pred, true), pred, label_num)
        fn_mask = torch.where(torch.ne(pred, true), true, label_num)

        tp = torch.bincount(tp_mask, minlength=self.rel_size + 1)[:self.rel_size]
        fp = torch.bincount(fp_mask, minlength=self.rel_size + 1)[:self.rel_size]
        fn = torch.bincount(fn_mask, minlength=self.rel_size + 1)[:self.rel_size]
        tn = torch.sum(mask_t & mask_p)
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def forward(self, batch):
        """
        Network Forward computation.
        Args:
            batch: dictionary with tensors
        Returns: (Tensors) loss, statistics, predictions, index
        """
        # Word Embeddings
        word_vec = self.input_layer(batch['words'])

        # Encoder
        encoded_seq = self.encoding_layer(word_vec, batch['word_sec'])

        # Graph
        graph, pindex, nodes_info, mask = self.graph_layer(encoded_seq, batch['entities'], batch['word_sec'],
                                                           batch['section'], batch['distances'])

        # Inference/Walks
        if self.walks_iter and self.walks_iter > 0:
            graph = self.walk(graph, adj_=batch['adjacency'], mask_=mask)

        # Classification
        select = self.select_pairs(batch['pairs4class'], nodes_info, pindex)
        graph = self.classifier(graph[select])

        loss, stats, preds = self.estimate_loss(graph, batch['relations'][select].long())
        return loss, stats, preds, select
