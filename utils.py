import torch
import random, os
import numpy as np
import wandb
import json
from pathlib import Path

def seed_setting(seed_number):
    random.seed(seed_number)
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def setup_wandb(args, seed):
    run = wandb.init(
        project=args.project_name,
        name=args.experiment_name + f'_seed_{seed}',
        config=args 
    )
    return run


def load_raw_data(data_filepath, use_GNN):
    print('Loading data...')
    train_idx = torch.load(data_filepath+'train_idx.pt')
    valid_idx = torch.load(data_filepath+'valid_idx.pt')
    test_idx = torch.load(data_filepath+'test_idx.pt')
    user_text = json.load(open(data_filepath+'norm_user_text.json'))
    labels = torch.load(data_filepath+'labels.pt')
    if use_GNN:
        edge_index = torch.load(data_filepath+'edge_index.pt')
        edge_type = torch.load(data_filepath+'edge_type.pt')
        return {'train_idx': train_idx, 
                'valid_idx': valid_idx, 
                'test_idx': test_idx, 
                'user_text': user_text, 
                'labels': labels, 
                'edge_index': edge_index, 
                'edge_type': edge_type}
    else:
        return {'train_idx': train_idx, 
                'valid_idx': valid_idx, 
                'test_idx': test_idx, 
                'user_text': user_text, 
                'labels': labels}


def load_distilled_knowledge(from_which_model, intermediate_data_filepath, iter):
    if from_which_model == 'LM':
        embeddings = torch.load(intermediate_data_filepath / f'embeddings_iter_{iter}.pt')
        soft_labels = torch.load(intermediate_data_filepath / f'soft_labels_iter_{iter}.pt')
        return embeddings, soft_labels
    
    elif from_which_model == 'GNN':
       
        soft_labels = torch.load(intermediate_data_filepath / f'soft_labels_iter_{iter}.pt')
        return soft_labels

    elif from_which_model == 'MLP':
        soft_labels = torch.load(intermediate_data_filepath / f'soft_labels_iter_{iter}.pt')
        return soft_labels
    
    else:
        raise ValueError('"from_which_model" should be "LM", "GNN" or "MLP".')


def prepare_path(experiment_name):
    experiment_path = Path(experiment_name)
    ckpt_filepath = experiment_path / 'checkpoints'
    MLP_ckpt_filepath = ckpt_filepath / 'MLP'
    LM_ckpt_filepath = ckpt_filepath / 'LM'
    GNN_ckpt_filepath = ckpt_filepath / 'GNN'
    MLP_KD_ckpt_filepath = Path('MLP_KD')
    LM_prt_ckpt_filepath = ckpt_filepath / 'LM_pretrain'
    GNN_prt_ckpt_filepath = ckpt_filepath / 'GNN_pretrain'
    LM_prt_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    GNN_prt_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    LM_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    GNN_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    MLP_KD_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    MLP_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    
    LM_intermediate_data_filepath = experiment_path / 'intermediate' / 'LM'
    GNN_intermediate_data_filepath = experiment_path / 'intermediate' / 'GNN'
    MLP_intermediate_data_filepath = experiment_path / 'intermediate' / 'MLP'
    LM_intermediate_data_filepath.mkdir(exist_ok=True, parents=True)
    GNN_intermediate_data_filepath.mkdir(exist_ok=True, parents=True)
    MLP_intermediate_data_filepath.mkdir(exist_ok=True, parents=True)

    return LM_prt_ckpt_filepath, GNN_prt_ckpt_filepath, MLP_KD_ckpt_filepath, LM_ckpt_filepath, GNN_ckpt_filepath, MLP_ckpt_filepath, LM_intermediate_data_filepath, GNN_intermediate_data_filepath, MLP_intermediate_data_filepath


    
def reset_split(n_nodes, ratio):
    idx = torch.randperm(n_nodes)
    split = list(map(int, ratio.split(',')))
    train_ratio = split[0] / sum(split)
    valid_ratio = split[1] / sum(split)

    train_idx = idx[: int(train_ratio * n_nodes)]
    valid_idx = idx[int(train_ratio * n_nodes): int((train_ratio + valid_ratio) * n_nodes)]
    test_idx = idx[int((train_ratio + valid_ratio) * n_nodes):]
    return train_idx, valid_idx, test_idx

