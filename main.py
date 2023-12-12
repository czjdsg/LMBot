from parser_args import parser_args
from utils import *
from trainer import LM_Trainer, GNN_Trainer, MLP_Trainer
import torch.nn as nn

def main(args):
    
    for seed in list(map(int, args.seeds.strip().split(','))):
        
        seed_setting(seed)
        LM_prt_ckpt_filepath, GNN_prt_ckpt_filepath, MLP_KD_ckpt_filepath, LM_ckpt_filepath, GNN_ckpt_filepath, MLP_ckpt_filepath, LM_intermediate_data_filepath, GNN_intermediate_data_filepath, MLP_intermediate_data_filepath = prepare_path(args.experiment_name + f'_seed_{seed}')

        data = load_raw_data(args.raw_data_filepath, args.use_GNN)

        run = setup_wandb(args, seed)

        if args.reset_split != '-1':
            train_idx, valid_idx, test_idx = reset_split(len(data['user_text']), args.reset_split)
            data['train_idx'], data['valid_idx'], data['test_idx'] = train_idx, valid_idx, test_idx

        LMTrainer = LM_Trainer(
            model_name=args.LM_model,
            classifier_n_layers=args.LM_classifier_n_layers,
            classifier_hidden_dim=args.LM_classifier_hidden_dim,
            device=args.device,
            pretrain_epochs=args.LM_pretrain_epochs,
            optimizer_name=args.optimizer_LM,
            lr=args.lr_LM,
            weight_decay=args.weight_decay_LM,
            dropout=args.dropout,
            att_dropout=args.LM_att_dropout,
            lm_dropout=args.LM_dropout,
            warmup=args.warmup,
            label_smoothing_factor=args.label_smoothing_factor,
            pl_weight=args.alpha,
            max_length=args.max_length,
            batch_size=args.batch_size_LM,
            grad_accumulation=args.LM_accumulation,
            lm_epochs_per_iter=args.LM_epochs_per_iter,
            temperature=args.temperature,
            pl_ratio=args.pl_ratio_LM,
            intermediate_data_filepath=LM_intermediate_data_filepath,
            ckpt_filepath=LM_ckpt_filepath,
            pretrain_ckpt_filepath=LM_prt_ckpt_filepath,
            raw_data_filepath=args.raw_data_filepath,
            train_idx=data['train_idx'],
            valid_idx=data['valid_idx'],
            test_idx=data['test_idx'],
            hard_labels=data['labels'],
            user_seq=data['user_text'],
            run=run,
            eval_patience=args.LM_eval_patience,
            activation=args.activation
        )
        
        MLPTrainer = MLP_Trainer(
            device=args.device,
            optimizer_name=args.optimizer_MLP,
            lr=args.lr_MLP,
            weight_decay=args.weight_decay_MLP,
            dropout=args.MLP_dropout,
            pl_weight=args.gamma,
            batch_size=args.batch_size_MLP,
            n_layers=args.MLP_n_layers,
            hidden_dim=args.MLP_hidden_dim,
            activation=args.activation,
            glnn_epochs=args.MLP_KD_epochs,
            mlp_epochs_per_iter=args.MLP_epochs_per_iter,
            temperature=args.temperature,
            pl_ratio=args.pl_ratio_MLP,
            intermediate_data_filepath=MLP_intermediate_data_filepath,
            ckpt_filepath=MLP_ckpt_filepath,
            KD_ckpt_filepath=MLP_KD_ckpt_filepath,
            train_idx=data['train_idx'],
            valid_idx=data['valid_idx'],
            test_idx=data['test_idx'],
            hard_labels=data['labels'],
            run=run,
            seed=seed,
            use_gnn = args.use_GNN
        )


        LMTrainer.build_model()
        LMTrainer.pretrain()
        MLPTrainer.build_model()

        if args.use_GNN:
            GNNTrainer = GNN_Trainer(
                model_name=args.GNN_model,
                device=args.device,
                optimizer_name=args.optimizer_GNN,
                lr=args.lr_GNN,
                weight_decay=args.weight_decay_GNN,
                dropout=args.GNN_dropout,
                pl_weight=args.beta,
                batch_size=args.batch_size_GNN,
                gnn_n_layers=args.n_layers,
                n_relations=args.n_relations,
                activation=args.activation,
                gnn_epochs_per_iter=args.GNN_epochs_per_iter,
                temperature=args.temperature,
                pl_ratio=args.pl_ratio_GNN,
                intermediate_data_filepath=GNN_intermediate_data_filepath,
                ckpt_filepath=GNN_ckpt_filepath,
                pretrain_ckpt_filepath=GNN_prt_ckpt_filepath,
                train_idx=data['train_idx'],
                valid_idx=data['valid_idx'],
                test_idx=data['test_idx'],
                hard_labels=data['labels'],
                edge_index=data['edge_index'],
                edge_type=data['edge_type'],
                run=run,
                SimpleHGN_att_res=args.SimpleHGN_att_res,
                att_heads=args.att_heads,
                RGT_semantic_heads=args.RGT_semantic_heads,
                gnn_hidden_dim=args.hidden_dim,
                lm_name = args.LM_model
            )
            GNNTrainer.build_model()
            for iter in range(args.max_iters):
                print(f'------Iter: {iter}/{args.max_iters-1}------')

                embeddings_LM, soft_labels_LM = load_distilled_knowledge('LM', LM_intermediate_data_filepath, iter-1)
                flag = GNNTrainer.train(embeddings_LM, soft_labels_LM)
                GNNTrainer.infer(embeddings_LM)
                if flag:
                    print(f'Early stop by GNN at iter {iter}!')
                    break

                soft_labels_GNN = load_distilled_knowledge('GNN', GNN_intermediate_data_filepath, iter)
                flag = LMTrainer.train(soft_labels_GNN)
                LMTrainer.infer()
                if flag:
                    print(f'Early stop by LM at iter {iter}!')
                    break
            
            print(f'Best LM is iter {LMTrainer.best_iter} epoch {LMTrainer.best_epoch}!')
            LMTrainer.test()
            
            print(f'Best GNN is iter {GNNTrainer.best_iter} epoch {GNNTrainer.best_epoch}!')
            embeddings_LM = LMTrainer.load_embedding(GNNTrainer.best_iter-1)
            GNNTrainer.test(embeddings_LM)
            
            # soft_labels_GNN = GNNTrainer.load_soft_labels(GNNTrainer.best_iter)
            # MLPTrainer.KD_GLNN(embeddings_LM, soft_labels_GNN)
            
            GNNTrainer.save_results(args.experiment_name + f'_seed_{seed}/results_GNN.json')
            
            
        else:
            for iter in range(args.max_iters):
                print(f'------Iter: {iter}/{args.max_iters-1}------')

                embeddings_LM, soft_labels_LM = load_distilled_knowledge('LM', LM_intermediate_data_filepath, iter-1)
                flag = MLPTrainer.train(embeddings_LM, soft_labels_LM)
                MLPTrainer.infer(embeddings_LM)
                if flag:
                    print(f'Early stop by MLP at iter {iter}!')
                    break

                soft_labels_MLP = load_distilled_knowledge('MLP', MLP_intermediate_data_filepath, iter)
                flag = LMTrainer.train(soft_labels_MLP)
                LMTrainer.infer()
                if flag:
                    print(f'Early stop by LM at iter {iter}!')
                    break
            
            print(f'Best LM is iter {LMTrainer.best_iter} epoch {LMTrainer.best_epoch}!')
            LMTrainer.test()
            
            print(f'Best MLP is iter {MLPTrainer.best_iter} epoch {MLPTrainer.best_epoch}!')
            embeddings_LM = LMTrainer.load_embedding(MLPTrainer.best_iter-1)
            MLPTrainer.test(embeddings_LM)
            
            MLPTrainer.save_results(args.experiment_name + f'_seed_{seed}/results_MLP.json')
        LMTrainer.save_results(args.experiment_name + f'_seed_{seed}/results_LM.json')
        
if __name__ == '__main__':
    args = parser_args()
    main(args)
