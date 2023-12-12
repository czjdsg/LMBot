import torch.nn as nn
from torch_geometric.nn import RGCNConv, HGTConv
from SimpleHGN import SimpleHGNConv
import torch
from RGT import RGTLayer
from torch_geometric.nn.models import MLP


class RGCN(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['gnn_hidden_dim']
        self.n_layers = model_config['gnn_n_layers']
        self.convs = nn.ModuleList([])
        self.linear_in = nn.Linear(model_config['lm_input_dim'], self.hidden_dim)
  
        for i in range(self.n_layers):
            self.convs.append(RGCNConv(self.hidden_dim, self.hidden_dim, model_config['n_relations']))

        self.dropout = nn.Dropout(model_config['dropout'])
        
        self.activation_name = model_config['activation'].lower()
        if self.activation_name == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activation_name == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError('Please choose activation function from "leakyrelu", "relu" or "elu".')
        
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, 2)

    def forward(self, x, edge_index, edge_type):
        x = self.linear_in(x)
        x = self.dropout(x)
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_type)
            x = self.activation(x)
        x = self.linear_pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear_out(x)
    

class SimpleHGN(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['hidden_dim']
        self.n_layers = model_config['n_layers']
        self.heads = model_config['att_heads']
        self.edge_res = model_config['SimpleHGN_att_res']


        self.convs = nn.ModuleList([])
        for i in range(self.n_layers):
            if i == 0:
                self.convs.append(SimpleHGNConv(768, self.hidden_dim, model_config['n_relations'], 32,  beta=self.edge_res))
            elif i == self.n_layers - 1:
                self.convs.append(SimpleHGNConv(self.hidden_dim, self.hidden_dim, model_config['n_relations'], 32,  beta=self.edge_res, final_layer=True))
            else:
                self.convs.append(SimpleHGNConv(self.hidden_dim, self.hidden_dim, model_config['n_relations'], 32, beta=self.edge_res))

        self.dropout = nn.Dropout(model_config['dropout'])      
        
        self.activation_name = model_config['activation'].lower()
        if self.activation_name == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activation_name == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError('Please choose activation function from "leakyrelu", "relu" or "elu".')
        
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, 2)

    def forward(self, x, edge_index, edge_type):
        for i in range(self.n_layers):
            if i == 0:
                x, att_res = self.convs[i](x, edge_index, edge_type)
                x = self.dropout(x)
            else:
                x, att_res = self.convs[i](x, edge_index, edge_type, pre_alpha=att_res)
                x = self.dropout(x)
        x = self.linear_pool(x)
        x = self.dropout(x)
        x = self.activation(x)

        return self.linear_out(x)
    


class HGT(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['gnn_hidden_dim']
        self.n_layers = model_config['gnn_n_layers']
        self.heads = model_config['att_heads']
        self.metadata = (['user'], [('user', 'follower', 'user'), ('user', 'following', 'user')])
        self.mlp = MLP_Model(model_config)
        

        self.convs = nn.ModuleList([])
        for i in range(self.n_layers):
            self.convs.append(HGTConv(self.hidden_dim, self.hidden_dim, self.metadata, self.heads))

        self.dropout = nn.Dropout(model_config['dropout'])
        
        self.activation_name = model_config['activation'].lower()
        if self.activation_name == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activation_name == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError('Please choose activation function from "leakyrelu", "relu" or "elu".')
        
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, 2)

    def prepare_data_for_HGT(self, x, edge_index, edge_type):
        x_dict = {self.metadata[0][0]: x}
        edge_index_dict = {}
        for i in range(len(self.metadata[1])):

            edge_index_dict[self.metadata[1][i]] = edge_index[:, edge_type==i]
        return x_dict, edge_index_dict

    def forward(self, x1, x2, x3, edge_index, edge_type):
        x = self.mlp(x1, x2, x3)
        x = self.dropout(x)     
        x_dict, edge_index_dict = self.prepare_data_for_HGT(x, edge_index, edge_type)
        for i in range(self.n_layers):
            x = self.convs[i](x_dict, edge_index_dict)
            x[self.metadata[0][0]] = self.activation(self.dropout(x[self.metadata[0][0]]))
            
        x = self.linear_pool(x[self.metadata[0][0]])
        x = self.dropout(x)
        x = self.activation(x)

        return self.linear_out(x)
    


class RGT(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['gnn_hidden_dim']
        self.n_layers = model_config['gnn_n_layers']
        self.n_relations = model_config['n_relations']
        self.mlp = MLP_Model(model_config)
        self.convs = nn.ModuleList([])
        for i in range(self.n_layers):
            self.convs.append(RGTLayer(self.n_relations, self.hidden_dim, self.hidden_dim, model_config['att_heads'], model_config['RGT_semantic_heads'], dropout=model_config['dropout']))

        self.dropout = nn.Dropout(model_config['dropout'])
        
        self.activation_name = model_config['activation'].lower()
        if self.activation_name == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activation_name == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError('Please choose activation function from "leakyrelu", "relu" or "elu".')
        
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, 2)

    def prepare_data_for_RGT(self, x, edge_index, edge_type):
        edge_index_list = []
        for i in range(self.n_relations):
            edge_index_list.append(edge_index[:, edge_type==i])
        return x, edge_index_list
    
    def forward(self, LM_embedding, x_numerical, x_categorical, edge_index, edge_type):
        x = self.mlp(LM_embedding, x_numerical, x_categorical)
        x, edge_index_list = self.prepare_data_for_RGT(x, edge_index, edge_type)
        # x = self.input_norm(x)
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index_list)
            x = self.activation(x)
        x = self.linear_pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear_out(x)
    
    
    