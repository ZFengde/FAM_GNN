import torch as th
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np
import time
from .fam_gnn import Ante_generator

class gaussmf():
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def ante(self, x):
        return th.exp(-((x - self.mean)**2.) / (2 * self.sigma **2.))

class TSFuzzyLayer(nn.Module): # -> attention, truth_value
    def __init__(self):
        super(TSFuzzyLayer, self).__init__()
        self.rules_num = 9
        # 2 * 9
        self.sub_systems_mat = nn.Parameter(th.tensor([[-0.2, -0.1, -0.05, -0.05, -0.05, -0.01, -0.055, -0.01, 0], 
                                                        [0, -0.002, -0.001, -0.001, -0.001, -0.0002, 0.0008, -0.0002, 0]]), requires_grad=False)
        self.sub_systems_bias = nn.Parameter(th.tensor([1, 0.7, 0.275, 0.4, 0.25, 0.07, 0.2225, 0.07, 0]), requires_grad=False)
        self._init_rules()

    def edge_func(self, edges):
        # preprocessing
        vector = edges.dst['h'] - edges.src['h'] # edge_num, batch, input_dim
        x1, x2 = Ante_generator(vector)  # edge_num, batch, 2
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

        # 3 * 3 --> 9, membership degree
        truth_value = self.ante_process(x1, x2) # edge_num, batch, 9, as coeffient

        # stack x1, x2 together, which is rules_num, edge_num * batch, 2
        premises = th.stack((x1, x2), dim=2).view(-1, 2).float()
        
        '''
        TS fuzzy system process here, including several process:
        1. truth_value: represent the satisfy degree of different rules
        2. consequence: output of different different rules linear functions
        3. attention: weighted cosequence, which weighted by truth_value, can be understood as coupling degree
        '''

        consequence = th.matmul(premises, self.sub_systems_mat) + self.sub_systems_bias
        consequence = consequence.view(x1.shape[0], x1.shape[1], self.rules_num) # edge_num, batch, 9

        # normalized with respect to the truth_values
        attention = th.sum((truth_value * consequence), dim=2) / th.sum(truth_value, dim=2)
        attention = attention.unsqueeze(2) # edge_num, batch, 1, represent the coupling degree of the edge

        # Softmax the coupling according to the edge type,
        # for i in range(len(self.edge_sg_ID)):
        #     attention[self.edge_sg_ID[i]] = th.softmax(attention[self.edge_sg_ID[i]], dim=0)
        attention[self.edge_sg_ID[0]] = th.softmax(attention[self.edge_sg_ID[0]], dim=0)
        
        return {'attention': attention}

    def forward(self, g, feat, edge_sg_ID):
        g.srcdata['h'] = feat # 9, batch, input_dim 
        self.edge_sg_ID = edge_sg_ID
        
        g.apply_edges(self.edge_func)
        # g.update_all(self.edge_func, fn.sum('attention', 'h'))
        
        return g.edata['attention'] # 72, batch

    def ante_process(self, x1, x2):
        x1_s_level = self.x1_s.ante(x1)
        x1_m_level = self.x1_m.ante(x1)
        x1_l_level = self.x1_l.ante(x1)

        x2_s_level = self.x2_s.ante(x2)
        x2_m_level = self.x2_m.ante(x2)
        x2_l_level = self.x2_l.ante(x2)

        truth_values = th.stack((th.min(x1_s_level, x2_s_level),
                         th.min(x1_s_level, x2_m_level),
                         th.min(x1_s_level, x2_l_level), 
                         th.min(x1_m_level, x2_s_level), 
                         th.min(x1_m_level, x2_m_level), 
                         th.min(x1_m_level, x2_l_level),
                         th.min(x1_l_level, x2_s_level),
                         th.min(x1_l_level, x2_m_level),
                         th.min(x1_l_level, x2_l_level)), dim=2) 

        return truth_values

    def _init_rules(self):
        self.x1_s = gaussmf(0, 0.75) # mean and sigma
        self.x1_m = gaussmf(2, 0.75)
        self.x1_l = gaussmf(4, 0.75)

        self.x2_s = gaussmf(0, 30) # mean and sigma
        self.x2_m = gaussmf(90, 30) # mean and sigma
        self.x2_l = gaussmf(180, 30) # mean and sigma

class Temp_FAM_GNNLayer(nn.Module): # using antecedants to update node features
    def __init__(self, in_feat, out_feat, num_rels, num_ntypes):
        super(Temp_FAM_GNNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_ntypes = num_ntypes

        # message weight and bias
        self.weight = nn.Parameter(th.Tensor(self.num_rels, self.in_feat, self.out_feat))
        self.m_bias = nn.Parameter(th.Tensor(self.num_rels, self.out_feat))

        # self-loop weight and bias
        self.loop_weight = nn.Parameter(th.Tensor(self.num_ntypes, self.in_feat, self.out_feat))
        self.h_bias = nn.Parameter(th.Tensor(self.num_ntypes, 1, self.out_feat))

        # nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.m_bias)
        nn.init.zeros_(self.h_bias)

    def message_func(self, edges):
        # here msg = attention * (W_rel * src.h + b_rel)
        w = self.weight[edges.data['rel_type']] # edge_num, in * out
        m_bias = self.m_bias[edges.data['rel_type']].unsqueeze(1) # edge_num, 1, out_feat

        attention = edges.data['attention'] # edge_num, batch, 1
        raw_msg =  th.bmm(edges.src['h'], w) + m_bias # edge_num, batch, out =  edge_num, in, out * edge_num, batch, in
        # raw_msg =  th.bmm(edges.dst['h'] - edges.src['h'], w) + m_bias # edge_num, batch, out =  edge_num, in, out * edge_num, batch, in
        msg = attention * raw_msg # edge_num, batch, out_feat = edge_num, batch, 1 * edge_num, batch, out_feat
        
        # and here is no self-loop
        return {'msg': msg} # edge_num, batch, out_feat

    def forward(self, g, feat, etypes, ntypes, attention):
        with g.local_scope(): 
            # pass node features and etypes information
            g.ndata['h'] = feat # node_num, batch, input_dim 
            g.edata['rel_type'] = etypes 
            g.edata['attention'] = attention # edge_num, batch, 1

            # self-loop
            loop_weight = self.loop_weight[ntypes] # node_num, input_dim, output_dim = 6, 6, 10
            loop_bias = self.h_bias[ntypes]
            if g.ndata['h'].dim() == 2:
                x = g.ndata['h'].view(-1, 1, self.in_feat)
                loop_value = th.bmm(x, loop_weight) + loop_bias
            elif g.ndata['h'].dim() == 3:
                loop_value = th.bmm(g.ndata['h'], loop_weight) + loop_bias
                
            # message passing
            g.update_all(self.message_func, fn.sum('msg', 'h'))
            h = g.ndata['h'] + loop_value

            return h

class Temp_FAM_GNN(nn.Module):
    def __init__(self, input_dim, h_dim, out_dim, num_rels, num_ntypes):
        super(Temp_FAM_GNN, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_ntypes = num_ntypes

        self.ante_layer = TSFuzzyLayer()
        self.layer1 = Temp_FAM_GNNLayer(self.input_dim, self.h_dim, self.num_rels, self.num_ntypes)
        self.layer2 = Temp_FAM_GNNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_ntypes)

    def forward(self, g, feat, etypes, ntypes, edge_sg_ID):
        
        # Attention mechanism
        attention = self.ante_layer(g, feat, edge_sg_ID) 

        x = th.tanh(self.layer1(g, feat, etypes, ntypes, attention))
        x = th.tanh(self.layer2(g, x, etypes, ntypes, attention)) # node_num, batch, out_dim
        # here we take robot, target, and a compressed obstacle info
        x = th.stack((x[0], x[1], th.mean(x[2:], dim=0)), dim=0)
        return x

