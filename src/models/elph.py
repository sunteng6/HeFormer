
from time import time
import logging

import torch

import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops

from src.models.gnn import SIGN, SIGNEmbedding
from src.hashing import ElphHashes



from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GraphConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_init=False, use_act=True):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x


class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):
        # feature transformation
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            vs = source_input.reshape(-1, 1, self.out_channels)

        # normalize input
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
            attention = attention / normalizer

        final_output = attn_output.mean(dim=1)

        if output_attn:
            return final_output, attention
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class SGFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True,
                 trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True,
                 gnn_use_residual=True, gnn_use_act=True,
                 use_graph=True, graph_weight=0.8, aggregate='add'):
        super().__init__()
        self.trans_conv = TransConv(in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout,
                                    trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act)
        self.graph_conv = GraphConv(in_channels, hidden_channels, gnn_num_layers, gnn_dropout, gnn_use_bn,
                                    gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)
        self.use_graph = use_graph
        self.graph_weight = graph_weight

        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters()) if self.graph_conv is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, x, edge_index):
        x1 = self.trans_conv(x)
        if self.use_graph:
            x2 = self.graph_conv(x, edge_index)
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1
        x = self.fc(x)
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()

class LinkPredictor(torch.nn.Module):
    def __init__(self, args, use_embedding=False):
        super(LinkPredictor, self).__init__()
        self.use_embedding = use_embedding
        self.use_feature = args.use_feature
        self.feature_dropout = args.feature_dropout
        self.label_dropout = args.label_dropout
        self.dim = args.max_hash_hops * (args.max_hash_hops + 2)
        self.label_lin_layer = Linear(self.dim, self.dim)
        if args.use_feature:
            self.bn_feats = torch.nn.BatchNorm1d(args.hidden_channels)
        if self.use_embedding:
            self.bn_embs = torch.nn.BatchNorm1d(args.hidden_channels)
        self.bn_labels = torch.nn.BatchNorm1d(self.dim)
        if args.use_feature:
            self.lin_feat = Linear(args.hidden_channels,
                                   args.hidden_channels)
            self.lin_out = Linear(args.hidden_channels, args.hidden_channels)
        out_channels = self.dim + args.hidden_channels if self.use_feature else self.dim
        if self.use_embedding:
            self.lin_emb = Linear(args.hidden_channels,
                                  args.hidden_channels)
            self.lin_emb_out = Linear(args.hidden_channels, args.hidden_channels)
            out_channels += args.hidden_channels
        self.lin = Linear(out_channels, 1)

    def feature_forward(self, x):
        """
        small neural network applied edgewise to hadamard product of node features
        @param x: node features torch tensor [batch_size, 2, hidden_dim]
        @return: torch tensor [batch_size, hidden_dim]
        """
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_out(x)
        x = self.bn_feats(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)
        return x

    def embedding_forward(self, x):
        x = self.lin_emb(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_emb_out(x)
        x = self.bn_embs(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)

        return x

    def forward(self, sf, node_features, emb=None):
        sf = self.label_lin_layer(sf)
        sf = self.bn_labels(sf)
        sf = F.relu(sf)
        x = F.dropout(sf, p=self.label_dropout, training=self.training)
        # process node features
        if self.use_feature:
            node_features = self.feature_forward(node_features)
            x = torch.cat([x, node_features.to(torch.float)], 1)
        if emb is not None:
            node_embedding = self.embedding_forward(emb)
            x = torch.cat([x, node_embedding.to(torch.float)], 1)
        x = self.lin(x)
        return x

    def print_params(self):
        print(f'model bias: {self.lin.bias.item():.3f}')
        print('model weights')
        for idx, elem in enumerate(self.lin.weight.squeeze()):
            if idx < self.dim:
                print(f'{self.idx_to_dst[idx % self.emb_dim]}: {elem.item():.3f}')
            else:
                print(f'feature {idx - self.dim}: {elem.item():.3f}')


class ELPH(torch.nn.Module):
    """
    propagating hashes, features and degrees with message passing
    """

    def __init__(self, args, num_features, node_embedding=None):
        super(ELPH, self).__init__()
        # hashing things
        self.elph_hashes = ElphHashes(args)
        self.init_hashes = None
        self.init_hll = None
        self.num_perm = args.minhash_num_perm
        self.hll_size = 2 ^ args.hll_p
        # gnn things
        self.use_feature = args.use_feature
        self.feature_prop = args.feature_prop  # None, residual, cat
        self.node_embedding = node_embedding
        self.propagate_embeddings = args.propagate_embeddings
        self.sign_k = args.sign_k
        self.label_dropout = args.label_dropout
        self.feature_dropout = args.feature_dropout
        self.num_layers = args.max_hash_hops
        self.dim = args.max_hash_hops * (args.max_hash_hops + 2)
        # construct the nodewise NN components
        self._convolution_builder(num_features, args.hidden_channels,
                                  args)  # build the convolutions for features and embs
        # construct the edgewise NN components
        self.predictor = LinkPredictor(args, node_embedding is not None)
        if self.sign_k != 0 and self.propagate_embeddings:
            # only used for the ddi where nodes have no features and transductive node embeddings are needed
            self.sign_embedding = SIGNEmbedding(args.hidden_channels, args.hidden_channels, args.hidden_channels,
                                                args.sign_k, args.sign_dropout)

    def _convolution_builder(self, num_features, hidden_channels, args):
        self.convs = torch.nn.ModuleList()
        if args.feature_prop in {'residual', 'cat'}:  # use a linear encoder
            self.feature_encoder = Linear(num_features, hidden_channels)
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        else:
            self.convs.append(
                GCNConv(num_features, hidden_channels))
        for _ in range(self.num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        if self.node_embedding is not None:
            self.emb_convs = torch.nn.ModuleList()
            for _ in range(self.num_layers):  # assuming the embedding has hidden_channels dims
                self.emb_convs.append(GCNConv(hidden_channels, hidden_channels))

    def propagate_embeddings_func(self, edge_index):
        num_nodes = self.node_embedding.num_embeddings
        gcn_edge_index, _ = gcn_norm(edge_index, num_nodes=num_nodes)
        return self.sign_embedding(self.node_embedding.weight, gcn_edge_index, num_nodes)

    def feature_conv(self, x, edge_index, k):
        if not self.use_feature:
            return None
        out = self.convs[k - 1](x, edge_index)
        out = F.dropout(out, p=self.feature_dropout, training=self.training)
        if self.feature_prop == 'residual':
            out = x + out
        return out

    def embedding_conv(self, x, edge_index, k):
        if x is None:
            return x
        out = self.emb_convs[k - 1](x, edge_index)
        out = F.dropout(out, p=self.feature_dropout, training=self.training)
        if self.feature_prop == 'residual':
            out = x + out
        return out

    def _encode_features(self, x):
        if self.use_feature:
            x = self.feature_encoder(x)
            x = F.dropout(x, p=self.feature_dropout, training=self.training)
        else:
            x = None

        return x

    def forward(self, x, edge_index):
        """
        @param x: raw node features tensor [n_nodes, n_features]
        @param adj_t: edge index tensor [2, num_links]
        @return:
        """
        hash_edge_index, _ = add_self_loops(edge_index)  # unnormalised, but with self-loops
        # if this is the first call then initialise the minhashes and hlls - these need to be the same for every model call
        num_nodes, num_features = x.shape
        if self.init_hashes == None:
            self.init_hashes = self.elph_hashes.initialise_minhash(num_nodes).to(x.device)
        if self.init_hll == None:
            self.init_hll = self.elph_hashes.initialise_hll(num_nodes).to(x.device)
        # initialise data tensors for storing k-hop hashes
        cards = torch.zeros((num_nodes, self.num_layers))
        node_hashings_table = {}
        for k in range(self.num_layers + 1):
            logger.info(f"Calculating hop {k} hashes")
            node_hashings_table[k] = {
                'hll': torch.zeros((num_nodes, self.hll_size), dtype=torch.int8, device=edge_index.device),
                'minhash': torch.zeros((num_nodes, self.num_perm), dtype=torch.int64, device=edge_index.device)}
            start = time()
            if k == 0:
                node_hashings_table[k]['minhash'] = self.init_hashes
                node_hashings_table[k]['hll'] = self.init_hll
                if self.feature_prop in {'residual', 'cat'}:  # need to get features to the hidden dim
                    x = self._encode_features(x)

            else:
                node_hashings_table[k]['hll'] = self.elph_hashes.hll_prop(node_hashings_table[k - 1]['hll'],
                                                                          hash_edge_index)
                node_hashings_table[k]['minhash'] = self.elph_hashes.minhash_prop(node_hashings_table[k - 1]['minhash'],
                                                                                  hash_edge_index)
                cards[:, k - 1] = self.elph_hashes.hll_count(node_hashings_table[k]['hll'])
                x = self.feature_conv(x, edge_index, k)

            logger.info(f'{k} hop hash generation ran in {time() - start} s')

        return x, node_hashings_table, cards


class HeFormer(torch.nn.Module):
    """
    Scalable version of ElPH that uses precomputation of subgraph features and SIGN style propagation
    of node features
    """

    def __init__(self, args, num_features=None, node_embedding=None):
        super(HeFormer, self).__init__()

        self.use_feature = args.use_feature
        self.label_dropout = args.label_dropout
        self.feature_dropout = args.feature_dropout
        self.node_embedding = node_embedding
        self.propagate_embeddings = args.propagate_embeddings
        # using both unormalised and degree normalised counts as features, hence * 2
        self.append_normalised = args.add_normed_features
        ra_counter = 1 if args.use_RA else 0
        num_labelling_features = args.max_hash_hops * (args.max_hash_hops + 2)
        self.dim = num_labelling_features * 2 if self.append_normalised else num_labelling_features
        self.use_RA = args.use_RA
        self.sign_k = args.sign_k
        if self.sign_k != 0:
            if self.propagate_embeddings:
                # this is only used for the ddi dataset where nodes have no features and transductive node embeddings are needed
                self.sign_embedding = SIGNEmbedding(args.hidden_channels, args.hidden_channels, args.hidden_channels,
                                                    args.sign_k, args.sign_dropout)
            else:
                self.sign = SIGN(num_features, args.hidden_channels, args.hidden_channels, args.sign_k,
                                 args.sign_dropout)
        self.label_lin_layer = Linear(self.dim, self.dim)
        if args.use_feature:
            self.bn_feats = torch.nn.BatchNorm1d(args.hidden_channels)
        if self.node_embedding is not None:
            self.bn_embs = torch.nn.BatchNorm1d(args.hidden_channels)
        self.bn_labels = torch.nn.BatchNorm1d(self.dim)
        self.bn_RA = torch.nn.BatchNorm1d(1)

        if args.use_feature:
            self.lin_feat = Linear(num_features,
                                   args.hidden_channels)
            self.lin_out = Linear(args.hidden_channels, args.hidden_channels)
        hidden_channels = self.dim + args.hidden_channels if self.use_feature else self.dim
        if self.node_embedding is not None:
            self.lin_emb = Linear(args.hidden_channels,
                                  args.hidden_channels)
            self.lin_emb_out = Linear(args.hidden_channels, args.hidden_channels)
            hidden_channels += self.node_embedding.embedding_dim

        self.Former = TransConv(in_channels=hidden_channels + ra_counter, hidden_channels=args.former_hidden, num_layers=args.former_num_layer,
                                num_heads=args.former_num_heads,
                                dropout=args.former_dropout, use_bn=True, use_residual=True, use_weight=True, use_act=True)

        self.GraphFormer = SGFormer(in_channels=hidden_channels + ra_counter, hidden_channels=1024, out_channels=1,
                                    trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True,
                                    trans_use_residual=True, trans_use_weight=True, trans_use_act=True,
                                    gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False,
                                    gnn_use_bn=True, gnn_use_residual=True, gnn_use_act=True,
                                    use_graph=True, graph_weight=0.8, aggregate='cat')

        self.lin = Linear(args.former_hidden, 1)

    def propagate_embeddings_func(self, edge_index):
        num_nodes = self.node_embedding.num_embeddings
        gcn_edge_index, _ = gcn_norm(edge_index, num_nodes=num_nodes)
        return self.sign_embedding(self.node_embedding.weight, gcn_edge_index, num_nodes)

    def _append_degree_normalised(self, x, src_degree, dst_degree):
        """
        Create a set of features that have the spirit of a cosine similarity x.y / ||x||.||y||. Some nodes (particularly negative samples)
        have zero degree
        because part of the graph is held back as train / val supervision edges and so divide by zero needs to be handled.
        Note that we always divide by the src and dst node's degrees for every node in ¬the subgraph
        @param x: unormalised features - equivalent to x.y
        @param src_degree: equivalent to sum_i x_i^2 as x_i in (0,1)
        @param dst_degree: equivalent to sum_i y_i^2 as y_i in (0,1)
        @return:
        """
        # this doesn't quite work with edge weights as instead of degree, the sum_i w_i^2 is needed,
        # but it probably doesn't matter
        normaliser = torch.sqrt(src_degree * dst_degree)
        normed_x = torch.divide(x, normaliser.unsqueeze(dim=1))
        normed_x[torch.isnan(normed_x)] = 0
        normed_x[torch.isinf(normed_x)] = 0
        return torch.cat([x, normed_x], dim=1)

    def feature_forward(self, x):
        """
        small neural network applied edgewise to hadamard product of node features
        @param x: node features torch tensor [batch_size, 2, hidden_dim]
        @return: torch tensor [batch_size, hidden_dim]
        """
        if self.sign_k != 0:
            x = self.sign(x)
        else:
            x = self.lin_feat(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_out(x)
        x = self.bn_feats(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)
        return x

    def embedding_forward(self, x):
        x = self.lin_emb(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_emb_out(x)
        x = self.bn_embs(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)

        return x

    def forward(self, sf, node_features, src_degree=None, dst_degree=None, RA=None, emb=None):
        """
        forward pass for one batch of edges
        @param sf: subgraph features [batch_size, num_hops*(num_hops+2)]
        @param node_features: raw node features [batch_size, 2, num_features]
        @param src_degree: degree of source nodes in batch
        @param dst_degree:
        @param RA:
        @param emb:
        @return:
        """
        if self.append_normalised:
            sf = self._append_degree_normalised(sf, src_degree, dst_degree)
        x = self.label_lin_layer(sf)
        x = self.bn_labels(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.label_dropout, training=self.training)  # (4096,8)
        if self.use_feature:
            node_features = self.feature_forward(node_features)
            x = torch.cat([x, node_features.to(torch.float)], 1)  # (4096,1032)
        if self.node_embedding is not None:
            node_embedding = self.embedding_forward(emb)
            x = torch.cat([x, node_embedding.to(torch.float)], 1)
        if self.use_RA:
            RA = RA.unsqueeze(-1)
            RA = self.bn_RA(RA)
            x = torch.cat([x, RA], 1)
        x = self.Former(x)

        x = self.lin(x)
        return x

    def print_params(self):
        print(f'model bias: {self.lin.bias.item():.3f}')
        print('model weights')
        for idx, elem in enumerate(self.lin.weight.squeeze()):
            if idx < self.dim:
                print(f'{self.idx_to_dst[idx % self.emb_dim]}: {elem.item():.3f}')
            else:
                print(f'feature {idx - self.dim}: {elem.item():.3f}')
