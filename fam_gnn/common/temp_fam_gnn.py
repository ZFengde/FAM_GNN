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
        attention[self.priority] = 1.
        
        return {'attention': attention}

    def forward(self, g, feat, etypes):
        g.srcdata['h'] = feat # 9, batch, input_dim 
        self.priority = th.cat((th.where(etypes==0)[0], th.where(etypes==4)[0]))
        
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
    def __init__(self, input_dim, h_dim, out_dim, num_rels, num_ntypes, node_num_pertime):
        super(Temp_FAM_GNN, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_ntypes = num_ntypes
        self.node_num_pertime = node_num_pertime

        self.ante_layer = TSFuzzyLayer()
        self.layer1 = Temp_FAM_GNNLayer(self.input_dim, self.h_dim, self.num_rels, self.num_ntypes)
        self.layer2 = Temp_FAM_GNNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_ntypes)

    def forward(self, g, feat, etypes, ntypes):
        # Attention mechanism
        attention = self.ante_layer(g, feat, etypes) 

        x = th.tanh(self.layer1(g, feat, etypes, ntypes, attention))
        x = th.tanh(self.layer2(g, x, etypes, ntypes, attention)) # node_num, batch, out_dim
        # here we take robot, target, and a compressed obstacle info
        x = th.split(x, self.node_num_pertime) # 3 * (7, 4, 8)
        x = th.mean(th.stack(x), dim=0)
        # x = th.stack((x[0], x[1], th.max(th.softmax(x[2:], dim=0), dim=0).values, th.min(th.softmax(x[2:], dim=0), dim=0).values, th.mean(th.softmax(x[2:], dim=0), dim=0)), dim=0)
        # x = x[0].unsqueeze(0)
        # x = th.stack((x[0], x[1]), dim=0)
        return x

class Temp_FAM_RelGraphConv(nn.Module):
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

class Temp_FAM_Rel_GCN(nn.Module):
    def __init__(self, input_dim, h_dim, out_dim, num_rels, node_num_pertime):
        super(Temp_FAM_Rel_GCN, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.node_num_pertime = node_num_pertime

        self.fam_layer = TSFuzzyLayer()
        self.layer1 = Temp_FAM_RelGraphConv(self.input_dim, self.h_dim, self.num_rels)
        self.layer2 = Temp_FAM_RelGraphConv(self.h_dim, self.out_dim, self.num_rels)

    def forward(self, g, feat, etypes):
        attention = self.fam_layer(g, feat, etypes) 
        x = th.tanh(self.layer1(g, feat, etypes, attention))
        x = th.tanh(self.layer2(g, x, etypes, attention)) # node_num, batch, out_dim
        # here we take robot, target, and a compressed obstacle info
        x = th.split(x, self.node_num_pertime) # 3 * (7, 4, 8)
        x = th.mean(th.stack(x), dim=0)
        # x = th.stack((x[0], x[1], th.max(th.softmax(x[2:], dim=0), dim=0).values, th.min(th.softmax(x[2:], dim=0), dim=0).values, th.mean(th.softmax(x[2:], dim=0), dim=0)), dim=0)
        # x = x[0].unsqueeze(0)
        # x = th.stack((x[0], x[1]), dim=0)
        return x

def temp_graph_and_types(node_num): # -> graph, edge_types
    edge_src = []
    edge_dst = []
    edge_types = []
    ID_indicator = 0
    temp_edge_src = []
    temp_edge_dst = []
    temp_edge_types = []
    for i in range(node_num):
        temp_edge_src.append(i+node_num)
        temp_edge_src.append(i+node_num*2)
        temp_edge_dst.append(i)
        temp_edge_dst.append(i+node_num)
        temp_edge_types.append(4)
        temp_edge_types.append(4)
        for j in range(node_num):

            if i == j:
                continue
            edge_src.append(i)
            edge_dst.append(j)
            '''
            relationships: 
            0: robot-target, 1: robot-obstacle
            2: target-obstacle, 3:obstacle-obstacle
            '''
            # robot-target
            if (i==0 and j==1) or (i==1 and j==0):
                edge_types.append(0)

            # robot-obstacle
            elif (i==0 and 2<=j) or (2<=i and j==0):
                edge_types.append(1)

            # target-obstacle
            elif (i==1 and 2<=j) or (2<=i and j==1):
                edge_types.append(2)

            # obstacle-obstacle
            else:
                edge_types.append(3)
            
            ID_indicator += 1
	
	# here generate node types for temp graph
    node_types = th.zeros(node_num, dtype=th.long) + 2 
    node_types[0] = 0
    node_types[1] = 1
    node_types = node_types.repeat(3)
    
	# here generate edge types for temp graph
    edge_types = th.cat((th.tensor(edge_types).repeat(3), th.tensor(temp_edge_types)), dim=0)
	# here generate edge types for temp graph
    edge_src = th.cat((th.tensor(edge_src), th.tensor(edge_src) + node_num, th.tensor(edge_src) + node_num * 2, th.tensor(temp_edge_src)), dim=0)
    edge_dst = th.cat((th.tensor(edge_dst), th.tensor(edge_dst) + node_num, th.tensor(edge_dst) + node_num * 2, th.tensor(temp_edge_dst)), dim=0)

    return dgl.graph((edge_src, edge_dst)), edge_types, node_types

def temp_obs_to_feat(obs, temp_1, temp_2): # -> node_infos
    # obs_size = 6 + 2 + 3*2 = 22 14
    def node_info_generator(obs_t):
        m = th.nn.ZeroPad2d((0, 4, 0, 0))
        obs_num = int((obs_t.shape[1] - 8) / 2)
        robot_info = obs_t[:, :6]
        target_info = m(obs_t[:, 6: 8]) # 6, 6
        obstacle_infos = m(obs_t[:, 8:].view(-1, obs_num, 2))
        node_infos = th.cat((robot_info.unsqueeze(1), target_info.unsqueeze(1), obstacle_infos), dim=1)
        return node_infos

    obs_t = node_info_generator(obs)
    obs_t_1 = node_info_generator(temp_1)
    obs_t_2 = node_info_generator(temp_2)

    temp_node_infos = th.cat((obs_t, obs_t_1, obs_t_2), dim=1)
    return temp_node_infos