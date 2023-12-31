import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, num_head, out_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.num_head = num_head
        self.att_layers = nn.ModuleList()
        # multi-head attention
        for i in range(num_head):
            self.att_layers.append(
            nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False))
            )
       
    def forward(self, z, return_beta):
        w = self.att_layers[0](z).mean(0)                    
        beta = torch.softmax(w, dim=0)                 
        if return_beta==True:
            print(beta)
        beta = beta.expand((z.shape[0],) + beta.shape)
        output = (beta * z).sum(1)

        for i in range(1, self.num_head):
            w = self.att_layers[i](z).mean(0)
            beta = torch.softmax(w, dim=0)
            if return_beta == True:
                print(beta)
            beta = beta.expand((z.shape[0],) + beta.shape)
            temp = (beta * z).sum(1)
            output += temp 
        # print('pre_feature',pre_features.size())
        return output / self.num_head

class RGTLayer(nn.Module):
    def __init__(self, num_edge_type, in_size, out_size, layer_num_heads, semantic_head, dropout):
        super(RGTLayer, self).__init__()
        self.gated = nn.Sequential(
            nn.Linear(in_size + out_size, in_size),
            nn.Sigmoid()
        )

        self.activation = nn.ELU()
        self.gat_layers = nn.ModuleList()
        for i in range(int(num_edge_type)):
            self.gat_layers.append(TransformerConv(in_channels=in_size, out_channels=out_size, heads=layer_num_heads, dropout=dropout, concat=False))
        
        self.semantic_attention = SemanticAttention(in_size=out_size, num_head = semantic_head, out_size = out_size)

    def forward(self, features, edge_index_list, beta = False, agg = None):
        
        u = self.gat_layers[0](features, edge_index_list[0].squeeze(0)).flatten(1) #.unsqueeze(1)
        a = self.gated(torch.cat((u, features), dim = 1))

        semantic_embeddings = (torch.mul(torch.tanh(u), a) + torch.mul(features, (1-a))).unsqueeze(1)
        
        for i in range(1,len(edge_index_list)):
            
            u = self.gat_layers[i](features, edge_index_list[i].squeeze(0)).flatten(1)
            a = self.gated(torch.cat((u, features), dim = 1))
            output = torch.mul(torch.tanh(u), a) + torch.mul(features, (1-a))
            semantic_embeddings=torch.cat((semantic_embeddings,output.unsqueeze(1)), dim = 1)
            
        if agg == 'max': 
            return semantic_embeddings.max(dim = 1)[0]
        if agg == 'min':
            return semantic_embeddings.min(dim = 1)[0]
        if agg == 'sum':
            return semantic_embeddings.sum(1)
        if agg == 'mean':
            return semantic_embeddings.mean(1)
        else:
            return self.semantic_attention(semantic_embeddings, return_beta = beta)