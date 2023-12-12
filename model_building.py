from LM import LM_Model
from GNNs import RGCN, HGT, SimpleHGN, RGT
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer

import torch


def build_LM_model(model_config):
    # build LM_model
    LM_model = LM_Model(model_config).to(model_config['device'])
    # bulid tokenizer
    LM_model_name = model_config['lm_model'].lower()

    if LM_model_name == 'deberta':
        LM_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    elif LM_model_name == 'roberta-f':
        LM_tokenizer = AutoTokenizer.from_pretrained('yzxjb/roberta-finetuned-20')
    elif LM_model_name == 'roberta':
        LM_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    elif LM_model_name == 'bert':
        LM_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif LM_model_name == 'twhin-bert':
        LM_tokenizer = AutoTokenizer.from_pretrained('Twitter/twhin-bert-base')
    elif LM_model_name == 'xlm-roberta':
        LM_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    if LM_model_name != 'roberta-f':
        special_tokens_dict = {'additional_special_tokens': ['DESCRIPTION:','METADATA:','TWEET:']}
        LM_tokenizer.add_special_tokens(special_tokens_dict)
        tokens_list = ["@USER", '#HASHTAG', "HTTPURL", 'EMOJI', 'RT', 'None']
        LM_tokenizer.add_tokens(tokens_list)
        LM_model.LM.resize_token_embeddings(len(LM_tokenizer))
    print('Information about LM model:')
    print('total params:', sum(p.numel() for p in LM_model.parameters()))
    return LM_model, LM_tokenizer


def build_GNN_model(model_config):
    # build GNN_model
    GNN_model_name = model_config['GNN_model'].lower()
    if GNN_model_name == 'rgcn':
        GNN_model = RGCN(model_config).to(model_config['device'])
    elif GNN_model_name == 'rgt':
        GNN_model = RGT(model_config).to(model_config['device'])
    elif GNN_model_name == 'simplehgn':
        GNN_model = SimpleHGN(model_config).to(model_config['device'])
    elif GNN_model_name == 'hgt':
        GNN_model = HGT(model_config).to(model_config['device'])
        
    else:
        raise ValueError('')

    print('Information about GNN model:')
    print(GNN_model)
    print('total params:', sum(p.numel() for p in GNN_model.parameters()))
    

    return GNN_model


