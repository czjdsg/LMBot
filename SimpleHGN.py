# from torch_geometric.nn import MessagePassing
# import torch
# import torch.nn as nn
# from torch.nn import Linear
# from torch_geometric.utils import softmax
# import torch.nn.functional as F

 

# class SimpleHGNConv(MessagePassing):
#     def __init__(self, in_dim, hidden_dim, num_edge_type, edge_emb_dim, num_heads, beta=0, is_final=False):
#         '''
#             if self.is_final:
#                 out_dim = hidden_dim * num_heads
#             else:
#                 out_dim = hidden_dim
#         '''
#         super(SimpleHGNConv, self).__init__(aggr='add')
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim 
#         self.beta = beta
#         self.num_heads = num_heads
#         self.is_final = is_final

#         self.W = nn.Parameter(torch.empty((in_dim, num_heads*hidden_dim)))
#         self.W_r = nn.Parameter(torch.empty((edge_emb_dim, num_heads*hidden_dim)))
#         self.edge_emb = nn.Parameter(torch.empty((num_edge_type, edge_emb_dim)))
#         self.a = nn.Parameter(torch.empty((1, num_heads, 3*hidden_dim)))
#         self.W_res = nn.Parameter(torch.empty((in_dim, num_heads*hidden_dim)))
        
#         self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
#         self.elu = nn.ELU()
        
#         nn.init.xavier_uniform_(self.W, gain=1.414)
#         nn.init.xavier_uniform_(self.W_r, gain=1.414)
#         nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
#         nn.init.xavier_uniform_(self.a, gain=1.414)
#         nn.init.xavier_uniform_(self.W_res, gain=1.414)
        
#     def forward(self, x, edge_index, edge_tpye, att_res=None, node_res=None):
#         '''
#             x has shape (num_nodes, in_dim)
#             edge_index has shape (2, num_edges)
#             edge_tpye has shape (num_edges, )
#         '''
#         out = self.propagate(x=x, edge_index=edge_index, edge_tpye=edge_tpye, att_res=att_res)
#         if self.is_final:
#             out = out.view(-1, self.num_heads, self.hidden_dim)
#             out = self.elu(out.sum(dim=1) / self.num_heads)
#             out = F.normalize(out, dim=1) 
#         else:
#             if node_res is not None:
#                 out = self.elu(out + torch.matmul(node_res * self.W_res))
#         return out, self.att.detach()

#     def message(self, x_i, x_j, edge_tpye, att_res, index, ptr, size_i):
#         v = torch.matmul(x_j, self.W)
#         k = torch.matmul(x_j, self.W).view(-1, self.num_heads, self.hidden_dim)
#         q = torch.matmul(x_i, self.W).view(-1, self.num_heads, self.hidden_dim)
#         '''
#             q, k, v has shape (num_edges, num_heads, hidden_dim)
#         '''
#         # print(k.shape)
#         att = self.leakyrelu((self.a * torch.cat([q, k, torch.matmul(self.edge_emb[edge_tpye], self.W_r).view(-1, self.num_heads, self.hidden_dim)], dim=-1)).sum(dim=-1))
#         att = softmax(att, index, ptr, size_i)
#         # print(att.shape)
#         self.att = att
#         '''
#             att has shape (num_edges, num_heads)
#         '''
#         att = att if att_res is None else self.beta * att_res + (1 - self.beta) * att

#         out = att.repeat(1, self.hidden_dim) * v
#         # print(out.shape)
#         '''
#             out has shape (num_edges, num_heads*hidden_dim)
#         '''
#         return out

#     def update(self, aggr_out):
#         return aggr_out


import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F

class SimpleHGNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edge_type, rel_dim, beta=None, final_layer=False):
        super(SimpleHGNConv, self).__init__(aggr = "add", node_dim=0)
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.W_r = torch.nn.Linear(rel_dim, out_channels, bias=False)
        self.a = torch.nn.Linear(3*out_channels, 1, bias=False)
        self.W_res = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.rel_emb = torch.nn.Embedding(num_edge_type, rel_dim)
        self.beta = beta
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.ELU = torch.nn.ELU()
        self.final = final_layer
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                    
    def forward(self, x, edge_index, edge_type, pre_alpha=None):
        
        node_emb = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, pre_alpha=pre_alpha)
        output = node_emb + self.W_res(x)
        output = self.ELU(output)
        if self.final:
            output = F.normalize(output, dim=1)
            
        return output, self.alpha.detach()
      
    def message(self, x_i, x_j, edge_type, pre_alpha, index, ptr, size_i):
        out = self.W(x_j)
        rel_emb = self.rel_emb(edge_type)
        alpha = self.leaky_relu(self.a(torch.cat((self.W(x_i), self.W(x_j), self.W_r(rel_emb)), dim=1)))
        alpha = softmax(alpha, index, ptr, size_i)
        if pre_alpha is not None and self.beta is not None:
            self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
        else:
            self.alpha = alpha
        out = out * alpha.view(-1,1)
        return out

    def update(self, aggr_out):
        return aggr_out