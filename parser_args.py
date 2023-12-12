import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    # wandb argument
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--experiment_name', type=str)
    # dataset argument
    
    parser.add_argument('--batch_size_LM', type=int, default=32)
    parser.add_argument('--batch_size_GNN', type=int, default=300000)
    parser.add_argument('--batch_size_MLP', type=int, default=300000)
    parser.add_argument('--raw_data_filepath', type=str, default='./data/raw/')
    parser.add_argument('--is_processed', type=bool, default=True)
    parser.add_argument('--reset_split', type=str, default='1,1,8')

    # model argument
    
    parser.add_argument('--LM_model', type=str, default='roberta')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--optimizer_LM', type=str, default='adamw')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--LM_classifier_n_layers', type=int, default=2)
    parser.add_argument('--LM_classifier_hidden_dim', type=int, default=128)
    parser.add_argument('--LM_dropout', type=float, default=0.1)
    parser.add_argument('--LM_att_dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing_factor', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.6)

    parser.add_argument('--use_GNN', action='store_true')
    parser.add_argument('--GNN_model', type=str, default='rgcn')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_relations', type=int, default=2)
    parser.add_argument('--activation', type=str, default='leakyrelu')
    parser.add_argument('--optimizer_GNN', type=str, default='adamw')
    parser.add_argument('--GNN_dropout', type=float, default=0.4)
    parser.add_argument('--att_heads', type=int, default=8)
    parser.add_argument('--SimpleHGN_att_res', type=float, default=0.2)
    parser.add_argument('--RGT_semantic_heads', type=int, default=8)

    parser.add_argument('--MLP_n_layers', type=int, default=3)
    parser.add_argument('--MLP_hidden_dim', type=int, default=128)
    parser.add_argument('--optimizer_MLP', type=str, default='adamw')
    parser.add_argument('--MLP_dropout', type=float, default=0.4)



    # train evaluation test argument
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--LM_pretrain_epochs', type=float, default=5)
    parser.add_argument('--MLP_KD_epochs', type=int, default=300)
    parser.add_argument('--LM_eval_patience', type=int, default=20)
    parser.add_argument('--LM_accumulation', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--GNN_epochs_per_iter', type=int, default=200)
    parser.add_argument('--LM_epochs_per_iter', type=int, default=3)
    parser.add_argument('--MLP_epochs_per_iter', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=3)
    parser.add_argument('--pl_ratio_LM', type=float, default=0.5)
    parser.add_argument('--pl_ratio_GNN', type=float, default=0)
    parser.add_argument('--pl_ratio_MLP', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--gamma', type=float, default=0.7)

    parser.add_argument('--lr_LM', type=float, default=1e-5)
    parser.add_argument('--weight_decay_LM', type=float, default=0.01)

    parser.add_argument('--lr_GNN', type=float, default=5e-4)
    parser.add_argument('--weight_decay_GNN', type=float, default=1e-5)

    parser.add_argument('--lr_MLP', type=float, default=5e-4)
    parser.add_argument('--weight_decay_MLP', type=float, default=1e-5)


    args = parser.parse_args()
    return args