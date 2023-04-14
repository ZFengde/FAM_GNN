import torch as th
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np
import time
from dgl.nn import GATConv
from .fam_gnn import TSFuzzyLayer

class RelGraphConv(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels):
        super().__init__()
        
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels

        self.weight = nn.Parameter(th.Tensor(self.num_rels, self.in_feat, self.out_feat))
        self.h_bias = nn.Parameter(th.Tensor(out_feat))
        self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))

        nn.init.zeros_(self.h_bias)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain("relu"))

    def message(self, edges):

        w = self.weight[edges.data['etype']] # 6, 6, 10
        m = th.bmm(edges.src['h'], w) # 6, 4, 6

        return {"m": m}

    def forward(self, g, feat, etype):
        with g.local_scope():
            g.srcdata["h"] = feat
            g.edata["etype"] = etype
            # message passing
            g.update_all(self.message, fn.sum("m", "h"))
            # apply bias and activation
            h = g.dstdata["h"]
            h = h + self.h_bias
            h = h + feat[: g.num_dst_nodes()] @ self.loop_weight
            return h

class FAM_RelGraphConv(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels):
        super().__init__()
        
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels

        self.weight = nn.Parameter(th.Tensor(self.num_rels, self.in_feat, self.out_feat))
        self.h_bias = nn.Parameter(th.Tensor(out_feat))
        self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))

        nn.init.zeros_(self.h_bias)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain("relu"))

    def message(self, edges):

        attention = edges.data['attention']
        w = self.weight[edges.data['etype']] # 6, 6, 10
        m = (th.bmm(edges.src['h'], w) + self.h_bias) * attention # 6, 4, 6

        return {"m": m}

    def forward(self, g, feat, etype, attention):
        with g.local_scope():
            g.srcdata["h"] = feat
            g.edata["etype"] = etype
            g.edata["attention"] = attention

            # message passing
            g.update_all(self.message, fn.sum("m", "h"))
            # apply bias and activation
            h = g.dstdata["h"]
            h = h + feat[: g.num_dst_nodes()] @ self.loop_weight
            return h

class FAM_GATConv(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.weight = nn.Parameter(th.Tensor(self.in_feat, self.out_feat))
        self.h_bias = nn.Parameter(th.Tensor(out_feat))

        nn.init.zeros_(self.h_bias)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

    def message(self, edges):

        attention = edges.data['attention']
        m = (th.matmul(edges.src['h'], self.weight) + self.h_bias) * attention # 6, 4, 6

        return {"m": m}

    def forward(self, g, feat, attention):
        with g.local_scope():
            g.srcdata["h"] = feat
            g.edata["attention"] = attention

            # message passing
            g.update_all(self.message, fn.sum("m", "h"))
            # apply bias and activation
            h = g.dstdata["h"]
            return h

class GAT(nn.Module):
    def __init__(self, input_dim, h_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.layer1 = GATConv(self.input_dim, self.h_dim, self.num_heads)
        self.layer2 = GATConv(self.h_dim * self.num_heads, self.out_dim, self.num_heads)

    def forward(self, g, feat): # 6, 20480, 6
        x = self.layer1(g, feat).flatten(2) # 6, 3, 6 --> 6, 3, 24
        x = th.mean(self.layer2(g, x), dim=2) # 6, 3, 24 --> 6, 3, 24
        x = th.stack((x[0], x[1], th.max(x[2:], dim=0).values, th.min(x[2:], dim=0).values, th.mean(x[2:], dim=0)), dim=0)
        return x
    
class Rel_GCN(nn.Module):
    def __init__(self, input_dim, h_dim, out_dim, num_rels):
        super(Rel_GCN, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels # num_etype

        self.layer1 = RelGraphConv(self.input_dim, self.h_dim, self.num_rels)
        self.layer2 = RelGraphConv(self.h_dim, self.out_dim, self.num_rels)

    def forward(self, g, feat, etypes):
        x = th.tanh(self.layer1(g, feat, etypes))
        x = th.tanh(self.layer2(g, x, etypes)) # node_num, batch, out_dim
        x = th.stack((x[0], x[1], th.max(x[2:], dim=0).values, th.min(x[2:], dim=0).values, th.mean(x[2:], dim=0)), dim=0)
        return x

class FAM_GAT(nn.Module):
    def __init__(self, input_dim, h_dim, out_dim):
        super(FAM_GAT, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.fam_layer = TSFuzzyLayer()
        self.layer1 = FAM_GATConv(self.input_dim, self.h_dim)
        self.layer2 = FAM_GATConv(self.h_dim, self.out_dim)

    def forward(self, g, feat, etypes): # 6, 20480, 6
        attention = self.fam_layer(g, feat, etypes) 
        x = self.layer1(g, feat, attention).flatten(2) # 6, 3, 6 --> 6, 3, 24
        x = self.layer2(g, x, attention) # 6, 3, 24 --> 6, 3, 24
        x = th.stack((x[0], x[1], th.max(x[2:], dim=0).values, th.min(x[2:], dim=0).values, th.mean(x[2:], dim=0)), dim=0)
        return x
    
class FAM_Rel_GCN(nn.Module):
    def __init__(self, input_dim, h_dim, out_dim, num_rels):
        super(FAM_Rel_GCN, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels # num_etype

        self.fam_layer = TSFuzzyLayer()
        self.layer1 = FAM_RelGraphConv(self.input_dim, self.h_dim, self.num_rels)
        self.layer2 = FAM_RelGraphConv(self.h_dim, self.out_dim, self.num_rels)

    def forward(self, g, feat, etypes):
        attention = self.fam_layer(g, feat, etypes) 
        x = th.tanh(self.layer1(g, feat, etypes, attention))
        x = th.tanh(self.layer2(g, x, etypes, attention)) # node_num, batch, out_dim
        # x = th.stack((x[0], x[1], th.max(x[2:], dim=0).values, th.min(x[2:], dim=0).values, th.mean(x[2:], dim=0)), dim=0)
        # x = th.stack((x[0], x[1], th.max(th.softmax(x[2:], dim=0), dim=0).values, th.min(th.softmax(x[2:], dim=0), dim=0).values, th.mean(th.softmax(x[2:], dim=0), dim=0)), dim=0)
        # x = x[0].unsqueeze(0)
        # x = th.stack((x[0], x[1]), dim=0)
        return x   
     
# g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,5,0]))
# feat = th.rand(6, 4, 6)
# # feat = th.rand(6, 6)

# etype = th.tensor([0,1,2,0,1,2])
# ntype = th.tensor([0,1,1,0,1,1])

# # # GAT
# # model = GAT(6, 10, 8, 3)
# # output = model(g, feat) 

# # Rel_GCN
# # model = Rel_GCN(6, 10, 8, 4)
# # output = model(g, feat, etype) 

# # # AGNNConv
# model = AGNNConv()
# output = model(g, feat) 
