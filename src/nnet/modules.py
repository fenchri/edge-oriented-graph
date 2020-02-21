#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06-Mar-2019

author: fenia
"""

import torch
from torch import nn, torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EmbedLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout, ignore=None, freeze=False, pretrained=None, mapping=None):
        """
        Args:
            num_embeddings: (tensor) number of unique items
            embedding_dim: (int) dimensionality of vectors
            dropout: (float) dropout rate
            trainable: (bool) train or not
            pretrained: (dict) pretrained embeddings
            mapping: (dict) mapping of items to unique ids
        """
        super(EmbedLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.freeze = freeze

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=ignore)

        if pretrained:
            self.load_pretrained(pretrained, mapping)
        self.embedding.weight.requires_grad = not freeze

        self.drop = nn.Dropout(dropout)

    def load_pretrained(self, pretrained, mapping):
        """
        Args:
            weights: (dict) keys are words, values are vectors
            mapping: (dict) keys are words, values are unique ids

        Returns: updates the embedding matrix with pre-trained embeddings
        """
        for word in mapping.keys():
            if word in pretrained:
                self.embedding.weight.data[mapping[word], :] = torch.from_numpy(pretrained[word])
            elif word.lower() in pretrained:
                self.embedding.weight.data[mapping[word], :] = torch.from_numpy(pretrained[word.lower()])

        assert (self.embedding.weight[mapping['and']].to('cpu').data.numpy() == pretrained['and']).all(), \
            'ERROR: Embeddings not assigned'

    def forward(self, xs):
        """
        Args:
            xs: (tensor) batchsize x word_ids

        Returns: (tensor) batchsize x word_ids x dimensionality
        """
        embeds = self.embedding(xs)
        if self.drop.p > 0:
            embeds = self.drop(embeds)

        return embeds


class Encoder(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers, bidirectional, dropout):
        """
        Wrapper for LSTM encoder
        Args:
            input_size (int): the size of the input features
            rnn_size (int):
            num_layers (int):
            bidirectional (bool):
            dropout (float):
        Returns: outputs, last_outputs
        - **outputs** of shape `(batch, seq_len, hidden_size)`:
          tensor containing the output features `(h_t)`
          from the last layer of the LSTM, for each t.
        - **last_outputs** of shape `(batch, hidden_size)`:
          tensor containing the last output features
          from the last layer of the LSTM, for each t=seq_len.
        """
        super(Encoder, self).__init__()

        self.enc = nn.LSTM(input_size=input_size,
                           hidden_size=rnn_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)

        # the dropout "layer" for the output of the RNN
        self.drop = nn.Dropout(dropout)

        # define output feature size
        self.feature_size = rnn_size

        if bidirectional:
            self.feature_size *= 2

    @staticmethod
    def sort(lengths):
        sorted_len, sorted_idx = lengths.sort()  # indices that result in sorted sequence
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(lengths.size(0) - 1, 0, lengths.size(0)).long()  # for big-to-small

        return sorted_idx, original_idx, reverse_idx

    def forward(self, embeds, lengths, hidden=None):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            embs (tensor): word embeddings
            lengths (list): the lengths of each sentence
        Returns: the logits for each class
        """
        # sort sequence
        sorted_idx, original_idx, reverse_idx = self.sort(lengths)

        # pad - sort - pack
        embeds = nn.utils.rnn.pad_sequence(embeds, batch_first=True, padding_value=0)
        embeds = embeds[sorted_idx][reverse_idx]  # big-to-small
        packed = pack_padded_sequence(embeds, list(lengths[sorted_idx][reverse_idx].data), batch_first=True)

        self.enc.flatten_parameters()
        out_packed, _ = self.enc(packed, hidden)

        # unpack
        outputs, _ = pad_packed_sequence(out_packed, batch_first=True)

        # apply dropout to the outputs of the RNN
        outputs = self.drop(outputs)

        # unsort the list
        outputs = outputs[reverse_idx][original_idx][reverse_idx]
        return outputs


class Classifier(nn.Module):
    def __init__(self, in_size, out_size, dropout):
        """
        Args:
            in_size: input tensor dimensionality
            out_size: outpout tensor dimensionality
            dropout: dropout rate
        """
        super(Classifier, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.lin = nn.Linear(in_features=in_size,
                             out_features=out_size,
                             bias=True)

    def forward(self, xs):
        """
        Args:
            xs: (tensor) batchsize x * x features

        Returns: (tensor) batchsize x * x class_size
        """
        if self.drop.p > 0:
            xs = self.drop(xs)

        xs = self.lin(xs)
        return xs






