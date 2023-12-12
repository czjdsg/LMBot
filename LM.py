from transformers import AutoModelForMaskedLM, AutoModel
import torch.nn as nn
from torch_geometric.nn.models import MLP
import torch

class LM_Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.LM_model_name = model_config['lm_model']
        if self.LM_model_name == 'deberta':
            self.LM = AutoModel.from_pretrained('microsoft/deberta-v3-base')
        elif self.LM_model_name == 'roberta':
            self.LM = AutoModel.from_pretrained('roberta-base')
        elif self.LM_model_name == 'bert':
            self.LM = AutoModel.from_pretrained('bert-base-uncased')
        elif self.LM_model_name == 'twhin-bert':
            self.LM = AutoModel.from_pretrained('Twitter/twhin-bert-base')
        elif self.LM_model_name == 'xlm-roberta':
            self.LM = AutoModel.from_pretrained('xlm-roberta-base')
        else:
            raise ValueError()
        
        self.classifier = MLP(in_channels=self.LM.config.hidden_size, hidden_channels=model_config['classifier_hidden_dim'], out_channels=2, num_layers=model_config['classifier_n_layers'], act=model_config['activation'])

        self.LM.config.hidden_dropout_prob = model_config['lm_dropout']
        self.LM.attention_probs_dropout_prob = model_config['att_dropout']

    def forward(self, tokenized_tensors):
        out = self.LM(output_hidden_states=True, **tokenized_tensors)['hidden_states']
        embedding = out[-1].mean(dim=1)
        
        return embedding.detach(), self.classifier(embedding)