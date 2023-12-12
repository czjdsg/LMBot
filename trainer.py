import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from model_building import build_LM_model, build_GNN_model
from dataloader import build_LM_dataloader, build_GNN_dataloader, build_MLP_dataloader
import os
import json
from pathlib import Path
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn.models import MLP

class LM_Trainer:
    def __init__(
            self, 
            model_name, 
            classifier_n_layers,
            classifier_hidden_dim,
            device, 
            pretrain_epochs,
            optimizer_name,
            lr,
            weight_decay,
            dropout,
            att_dropout,
            lm_dropout,
            activation,
            warmup,
            label_smoothing_factor,
            pl_weight,
            max_length,
            batch_size,
            grad_accumulation,
            lm_epochs_per_iter,
            temperature,
            pl_ratio,
            eval_patience,
            intermediate_data_filepath,
            ckpt_filepath,
            pretrain_ckpt_filepath,
            raw_data_filepath,
            train_idx,
            valid_idx,
            test_idx,
            hard_labels,
            user_seq,
            run):
        
        self.model_name = model_name
        self.device = device
        self.pretrain_epochs = pretrain_epochs
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.lm_dropout = lm_dropout
        self.warmup = warmup
        self.label_smoothing_factor = label_smoothing_factor
        self.pl_weight = pl_weight
        self.max_length = max_length
        self.batch_size = batch_size
        self.grad_accumulation = grad_accumulation
        self.lm_epochs_per_iter = lm_epochs_per_iter
        self.temperature = temperature
        self.pl_ratio = pl_ratio
        self.eval_patience = eval_patience
        self.intermediate_data_filepath = intermediate_data_filepath    
        self.ckpt_filepath = ckpt_filepath
        self.pretrain_ckpt_filepath = pretrain_ckpt_filepath
        self.raw_data_filepath = Path(raw_data_filepath)
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.hard_labels = hard_labels
        self.user_seq = user_seq
        self.run = run
        self.do_mlm_task = False
        
        self.iter = 0
        self.best_iter = 0
        self.best_valid_acc = 0
        self.best_epoch = 0
        self.criterion = CrossEntropyLoss(label_smoothing=label_smoothing_factor)
        self.KD_criterion = KLDivLoss(log_target=False, reduction='batchmean')
        self.results = {}

        
        self.get_train_idx_all()
        self.pretrain_steps_per_epoch = self.train_idx.shape[0] // self.batch_size + 1
        self.pretrain_steps = int(self.pretrain_steps_per_epoch * self.pretrain_epochs)
        self.train_steps_per_iter = (self.train_idx_all.shape[0] // self.batch_size + 1) * self.lm_epochs_per_iter
        self.optimizer_args = dict(lr=lr, weight_decay=weight_decay)

        self.model_config = {
            'lm_model': model_name,
            'dropout': dropout,
            'att_dropout': att_dropout,
            'lm_dropout': self.lm_dropout,
            'classifier_n_layers': classifier_n_layers,
            'classifier_hidden_dim': classifier_hidden_dim,
            'activation': activation,
            'device': device,
            'return_mlm_loss': True if self.do_mlm_task else False
            }
        
        self.dataloader_config = {
            'batch_size': batch_size,
            'pl_ratio': pl_ratio
            }
        
    

        
    def build_model(self):
        self.model, self.tokenizer = build_LM_model(self.model_config)
        self.DESCRIPTION_id = self.tokenizer.convert_tokens_to_ids('DESCRIPTION:')
        self.TWEET_id = self.tokenizer.convert_tokens_to_ids('TWEET:')
        self.METADATA_id = self.tokenizer.convert_tokens_to_ids('METADATA:')
       
    def get_optimizer(self, parameters):
        
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(parameters, **self.optimizer_args)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, **self.optimizer_args)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(parameters, **self.optimizer_args)
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(parameters, **self.optimizer_args)
        else:
            return NotImplementedError
        
        return optimizer
    
    def get_scheduler(self, optimizer, mode='train'):
        if mode == 'pretrain':
            return get_cosine_schedule_with_warmup(optimizer, self.pretrain_steps_per_epoch * self.warmup, self.pretrain_steps) 
        else:
            return CosineAnnealingLR(optimizer, T_max=self.train_steps_per_iter, eta_min=0)
        
    def get_initial_embeddings(self):
        
        if not os.path.exists(self.raw_data_filepath / f'embeddings_{self.model_name.lower()}.pt'):
            print('Generating initial GNN embeddings...')
            self.infer(True)

        embeddings = torch.load(self.raw_data_filepath / f'embeddings_{self.model_name.lower()}.pt')
        return embeddings
    
    def pretrain(self):
        print('LM pretraining start!')
        optimizer = self.get_optimizer(self.model.parameters())
        scheduler = self.get_scheduler(optimizer, 'pretrain')
        if os.listdir(self.pretrain_ckpt_filepath) and os.path.exists(self.intermediate_data_filepath / 'embeddings_iter_-1.pt'):
            print('Pretrain checkpoint exists, loading from checkpoint...')
            print('Please make sure you use the same parameter setting as the one of the pretrain checkpoint!')
            ckpt = torch.load(self.pretrain_ckpt_filepath / os.listdir(self.pretrain_ckpt_filepath)[0])
            self.model.load_state_dict(ckpt['model'])
            # self.optimizer.load_state_dict(ckpt['optimizer'])
            # self.scheduler.load_state_dict(ckpt['scheduler'])
            # embeddings = torch.load(self.intermediate_data_filepath / 'embeddings_iter_-1.pt')
            test_acc, test_f1 = self.eval('test')
            self.results['pretrain accuracy'] = test_acc
            self.results['pretrain f1'] = test_f1
        
        else:
            step = 0
            valid_acc_best = 0
            valid_step_best = 0
            
            torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.pretrain_ckpt_filepath / 'best.pkl')
            
            train_loader = build_LM_dataloader(self.dataloader_config, self.train_idx, self.user_seq, self.hard_labels, 'pretrain')

            for epoch in range(int(self.pretrain_epochs)+1):
                self.model.train()
                print(f'------LM Pretraining Epoch: {epoch}/{int(self.pretrain_epochs)}------')
                for batch in tqdm(train_loader):
                    step += 1
                    if step >= self.pretrain_steps:
                        break
                    tokenized_tensors, labels, _ = self.batch_to_tensor(batch)

                    _, output = self.model(tokenized_tensors)
                    loss = self.criterion(output, labels)
                    loss /= self.grad_accumulation
                    loss.backward()
                    self.run.log({'LM Pretrain Loss': loss.item()})
                    
                    if step % self.grad_accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    scheduler.step()

                    if step % self.eval_patience == 0:
                        valid_acc, valid_f1 = self.eval()

                        print(f'LM Pretrain Valid Accuracy = {valid_acc}')
                        print(f'LM Pretrain Valid F1 = {valid_f1}')
                        self.run.log({'LM Pretrain Valid Accuracy': valid_acc})
                        self.run.log({'LM Pretrain Valid F1': valid_f1})

                        if valid_acc > valid_acc_best:
                            valid_acc_best = valid_acc
                            valid_step_best = step
                            
                            torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.pretrain_ckpt_filepath / 'best.pkl')
                    
            
            print(f'The highest pretrain valid accuracy is {valid_acc_best}!')
            print(f'Load model from step {valid_step_best}')
            self.model.eval()
            all_outputs = []
            all_labels = []
            embeddings = []
            infer_loader = build_LM_dataloader(self.dataloader_config, None, self.user_seq, self.hard_labels, mode='infer')
            with torch.no_grad():
                ckpt = torch.load(self.pretrain_ckpt_filepath / 'best.pkl')
                self.model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler.load_state_dict(ckpt['scheduler'])
                for batch in tqdm(infer_loader):
                    tokenized_tensors, labels, _ = self.batch_to_tensor(batch)
                    embedding, output = self.model(tokenized_tensors)
                    embeddings.append(embedding.cpu())
                    all_outputs.append(output.cpu())
                    all_labels.append(labels.cpu())
                
                all_outputs = torch.cat(all_outputs, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                embeddings = torch.cat(embeddings, dim=0)
                soft_labels = torch.softmax(all_outputs / self.temperature, dim=1)
                soft_labels[self.train_idx] = all_labels[self.train_idx]

                test_predictions = torch.argmax(all_outputs[self.test_idx], dim=1).numpy()
                test_labels = torch.argmax(all_labels[self.test_idx], dim=1).numpy()
                torch.save(embeddings, self.intermediate_data_filepath / 'embeddings_iter_-1.pt')
                torch.save(soft_labels, self.intermediate_data_filepath / 'soft_labels_iter_-1.pt')

                test_acc = accuracy_score(test_predictions, test_labels)
                test_f1 = f1_score(test_predictions, test_labels)
                self.results['pretrain accuracy'] = test_acc
                self.results['pretrain f1'] = test_f1
        

        print(f'LM Pretrain Test Accuracy = {test_acc}')
        print(f'LM Pretrain Test F1 = {test_f1}')
        self.run.log({'LM Pretrain Test Accuracy': test_acc})
        self.run.log({'LM Pretrain Test F1': test_f1})


        

    def train(self, soft_labels):
        for param in self.model.classifier.parameters():
            param.requires_grad = False
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.get_optimizer(parameters)
        scheduler = self.get_scheduler(optimizer)
        
        early_stop_flag = True
        print('LM training start!')
        step = 0
        train_loader = build_LM_dataloader(self.dataloader_config, self.train_idx_all, self.user_seq, soft_labels, 'train', self.is_pl)

        
        for epoch in range(self.lm_epochs_per_iter):
            self.model.train()
            print(f'This is iter {self.iter} epoch {epoch}/{self.lm_epochs_per_iter-1}')

            for batch in tqdm(train_loader):
                step += 1
                
                tokenized_tensors, labels, is_pl = self.batch_to_tensor(batch)

                _, output = self.model(tokenized_tensors)
                
                pl_idx = torch.nonzero(is_pl == 1).squeeze()
                rl_idx = torch.nonzero(is_pl == 0).squeeze()
 
                if pl_idx.numel() == 0:
                    loss = self.criterion(output[rl_idx], labels[rl_idx])
                elif rl_idx.numel() == 0:
                    loss = self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), labels[pl_idx])
                else:
                    loss_KD = self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), labels[pl_idx])
                    loss_H = self.criterion(output[rl_idx], labels[rl_idx])
                    self.run.log({'loss_KD': loss_KD.item()})
                    self.run.log({'loss_H': loss_H.item()})
                    loss = self.pl_weight * loss_KD + (1 - self.pl_weight) * loss_H
                
                loss /= self.grad_accumulation
                loss.backward()
                self.run.log({'LM Train Loss': loss.item()})
                
                if step % self.grad_accumulation == 0:
                    
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()
                if step % self.eval_patience == 0:
                    valid_acc, valid_f1 = self.eval()

                    print(f'LM Valid Accuracy = {valid_acc}')
                    print(f'LM Valid F1 = {valid_f1}')
                    self.run.log({'LM Valid Accuracy': valid_acc})
                    self.run.log({'LM Valid F1': valid_f1})

                    if valid_acc > self.best_valid_acc:
                        early_stop_flag = False
                        self.best_valid_acc = valid_acc
                        self.best_iter = self.iter
                        self.best_epoch = epoch
                        torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.ckpt_filepath / 'best.pkl')
                
        print(f'The highest valid accuracy is {self.best_valid_acc}!')
        return early_stop_flag
    
    def infer(self, provide_embeddings_for_GNN_pretraining=False):
        self.model.eval()
        infer_loader = build_LM_dataloader(self.dataloader_config, None, self.user_seq, self.hard_labels, mode='infer')
        all_outputs = []
        all_labels = []
        embeddings = []
        with torch.no_grad():
            if provide_embeddings_for_GNN_pretraining:
                for batch in tqdm(infer_loader):
                    tokenized_tensors, labels, _ = self.batch_to_tensor(batch)
                    embedding, _ = self.model(tokenized_tensors)
                    embeddings.append(embedding.cpu())
                embeddings = torch.cat(embeddings, dim=0)
                torch.save(embeddings, self.raw_data_filepath / f'embeddings_{self.model_name.lower()}.pt')
                
            else:
                ckpt = torch.load(self.ckpt_filepath / 'best.pkl')
                self.model.load_state_dict(ckpt['model'])
                # self.optimizer.load_state_dict(ckpt['optimizer'])
                # self.scheduler.load_state_dict(ckpt['scheduler'])
                for batch in tqdm(infer_loader):
                    tokenized_tensors, labels, _ = self.batch_to_tensor(batch)

                    embedding, output = self.model(tokenized_tensors)
                    embeddings.append(embedding.cpu())
                    all_outputs.append(output.cpu())
                    all_labels.append(labels.cpu())
                
                all_outputs = torch.cat(all_outputs, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                embeddings = torch.cat(embeddings, dim=0)
        
                soft_labels = torch.softmax(all_outputs / self.temperature, dim=1)
                soft_labels[self.train_idx] = all_labels[self.train_idx]

                torch.save(soft_labels, self.intermediate_data_filepath / f'soft_labels_iter_{self.iter}.pt')
            
                torch.save(embeddings, self.intermediate_data_filepath / f'embeddings_iter_{self.iter}.pt')

                self.iter += 1

    def eval(self, mode='valid'):
        if mode == 'valid':            
            eval_loader =  build_LM_dataloader(self.dataloader_config, self.valid_idx, self.user_seq, self.hard_labels, mode='eval')
        elif mode == 'test':
            eval_loader =  build_LM_dataloader(self.dataloader_config, self.test_idx, self.user_seq, self.hard_labels, mode='eval')
        self.model.eval()

        valid_predictions = []
        valid_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader):
                tokenized_tensors, labels, _ = self.batch_to_tensor(batch)

                _, output = self.model(tokenized_tensors)

                valid_predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                valid_labels.append(torch.argmax(labels, dim=1).cpu().numpy())

            valid_predictions = np.concatenate(valid_predictions)
            valid_labels = np.concatenate(valid_labels)
            valid_acc = accuracy_score(valid_labels, valid_predictions)
            valid_f1 = f1_score(valid_labels, valid_predictions)

            return valid_acc, valid_f1


    def test(self):
        print('Computing test accuracy and f1 for LM...')
        ckpt = torch.load(self.ckpt_filepath / 'best.pkl')
        self.model.load_state_dict(ckpt['model'])
        test_acc, test_f1 = self.eval('test')
        print(f'LM Test Accuracy = {test_acc}')
        print(f'LM Test F1 = {test_f1}')
        self.run.log({'LM Test Accuracy': test_acc})
        self.run.log({'LM Test F1': test_f1})
        self.results['accuracy'] = test_acc
        self.results['f1'] = test_f1

    def batch_to_tensor(self, batch):
                    
        tokenized_tensors = self.tokenizer(text=batch[0], return_tensors='pt', max_length=self.max_length, truncation=True, padding='longest', add_special_tokens=False)
        for key in tokenized_tensors.keys():
            tokenized_tensors[key] = tokenized_tensors[key].to(self.device)
        labels = batch[1].to(self.device)
    
        if len(batch) == 3:
            is_pl = batch[2].to(self.device)
            return tokenized_tensors, labels, is_pl
        else:
            return tokenized_tensors, labels, None
        
    def load_embedding(self, iter):
        embeddings = torch.load(self.intermediate_data_filepath / f'embeddings_iter_{iter}.pt')
        return embeddings
    
    def save_results(self, path):
        json.dump(self.results, open(path, 'w'), indent=4)
        
    def get_train_idx_all(self):
        n_total = self.hard_labels.shape[0]
        all = set(np.arange(n_total))
        exclude = set(self.train_idx.numpy())
        n = self.train_idx.shape[0]
        pl_ratio_LM = min(self.pl_ratio, (n_total - n) / n)
        n_pl_LM = int(n * pl_ratio_LM)
        pl_idx = torch.LongTensor(np.random.choice(np.array(list(all - exclude)), n_pl_LM, replace=False))
        self.train_idx_all = torch.cat((self.train_idx, pl_idx))
        self.is_pl = torch.ones_like(self.train_idx_all, dtype=torch.int64)
        self.is_pl[0: n] = 0


class GNN_Trainer:
    def __init__(
        self, 
        model_name, 
        device, 
        optimizer_name,
        lr,
        weight_decay,
        dropout,
        pl_weight,
        batch_size,
        gnn_n_layers,
        n_relations,
        activation,
        gnn_epochs_per_iter,
        temperature,
        pl_ratio,
        intermediate_data_filepath,
        ckpt_filepath,
        pretrain_ckpt_filepath,
        train_idx,
        valid_idx,
        test_idx,
        hard_labels,
        edge_index, 
        edge_type,
        run,
        SimpleHGN_att_res,
        att_heads,
        RGT_semantic_heads,
        gnn_hidden_dim,
        lm_name
        ):
    
        self.model_name = model_name
        self.device = device
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.pl_weight = pl_weight
        self.dropout = dropout        
        self.batch_size = batch_size
        self.gnn_n_layers = gnn_n_layers
        self.n_relations = n_relations
        self.activation = activation
        self.gnn_epochs_per_iter = gnn_epochs_per_iter
        self.temperature = temperature
        self.pl_ratio = pl_ratio
        self.intermediate_data_filepath = intermediate_data_filepath    
        self.ckpt_filepath = ckpt_filepath
        self.pretrain_ckpt_filepath = pretrain_ckpt_filepath
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.hard_labels = hard_labels
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.run = run
        self.SimpleHGN_att_res = SimpleHGN_att_res
        self.att_heads = att_heads
        self.RGT_semantic_heads = RGT_semantic_heads
        self.gnn_hidden_dim = gnn_hidden_dim
        self.lm_input_dim = 1024 if lm_name.lower() in ['roberta-large'] else 768
        self.iter = 0
        self.best_iter = 0
        self.best_valid_acc = 0
        self.best_valid_epoch = 0
        self.criterion = CrossEntropyLoss()
        self.KD_criterion = KLDivLoss(log_target=False, reduction='batchmean')

        
        self.results = {}
        self.get_train_idx_all()
        self.optimizer_args = dict(lr=lr, weight_decay=weight_decay)
        
        self.model_config = {
            'GNN_model': model_name,
            'optimizer': optimizer_name,
            'gnn_n_layers': gnn_n_layers,
            'n_relations': n_relations,
            'activation': activation,
            'dropout': dropout,
            'gnn_hidden_dim': gnn_hidden_dim,
            'lm_input_dim': self.lm_input_dim,
            'SimpleHGN_att_res': SimpleHGN_att_res,
            'att_heads': att_heads,
            'RGT_semantic_heads': RGT_semantic_heads,
            'device': device
            }
        
        self.dataloader_config = {
            'batch_size': batch_size,
            'n_layers': gnn_n_layers
            }
        
    

    def build_model(self):
        self.model = build_GNN_model(self.model_config)

    def get_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.gnn_epochs_per_iter, eta_min=0)
    

    def get_optimizer(self):
        
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(self.model.parameters(), **self.optimizer_args)
        else:
            return NotImplementedError
        
        return optimizer
   
    def train(self, embeddings_LM, soft_labels):
        early_stop_flag = True
        
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        print('GNN training start!')
        print(f'This is iter {self.iter}')

        train_loader = build_GNN_dataloader(self.dataloader_config, self.train_idx_all, embeddings_LM, soft_labels, self.edge_index, self.edge_type, mode='train', is_pl=self.is_pl)
        
        for epoch in tqdm(range(self.gnn_epochs_per_iter)):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                batch_size = batch.batch_size
                x_batch = batch.x.to(self.device)

                edge_index_batch = batch.edge_index.to(self.device)
                edge_type_batch = batch.edge_type.to(self.device)
                is_pl = batch.is_pl[0: batch_size].to(self.device)
                labels = batch.labels[0: batch_size].to(self.device)

                output = self.model(x_batch, edge_index_batch, edge_type_batch) 
                output = output[0: batch_size]

                pl_idx = torch.nonzero(is_pl == 1).squeeze()
                rl_idx = torch.nonzero(is_pl == 0).squeeze()

        
                if pl_idx.numel() == 0:
                    loss = self.criterion(output[rl_idx], labels[rl_idx])
                elif rl_idx.numel() == 0:
                    loss = self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), labels[pl_idx])
                else:
                    # loss = self.pl_weight * self.criterion(output[pl_idx], labels[pl_idx]) + (1 - self.pl_weight) * self.criterion(output[rl_idx], labels[rl_idx])
                    loss = self.pl_weight * self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), labels[pl_idx]) + (1 - self.pl_weight) * self.criterion(output[rl_idx], labels[rl_idx])

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.run.log({'GNN Train Loss': loss.item()})
            

            valid_acc, valid_f1 = self.eval(embeddings_LM)
     
            self.run.log({'GNN Valid Accuracy': valid_acc})
            self.run.log({'GNN Valid F1': valid_f1})

            if valid_acc > self.best_valid_acc:
                early_stop_flag = False
                self.best_valid_acc = valid_acc
                self.best_epoch = epoch
                self.best_iter = self.iter
                torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.ckpt_filepath / 'best.pkl')
        print(f'The highest valid accuracy is {self.best_valid_acc}!')
        return early_stop_flag
    
    def infer(self, embeddings_LM):
        self.model.eval()
        infer_loader = build_GNN_dataloader(self.dataloader_config, None, embeddings_LM, self.hard_labels, self.edge_index, self.edge_type, mode='infer')

        all_outputs = []
        all_labels = []
        with torch.no_grad():
            ckpt = torch.load(self.ckpt_filepath / 'best.pkl')
            self.model.load_state_dict(ckpt['model'])
            # self.optimizer.load_state_dict(ckpt['optimizer'])
            # self.scheduler.load_state_dict(ckpt['scheduler'])
            for batch in infer_loader:
                batch_size = batch.batch_size
                x_batch = batch.x.to(self.device)

                edge_index_batch = batch.edge_index.to(self.device)
                edge_type_batch = batch.edge_type.to(self.device)
                labels = batch.labels[0: batch_size].to(self.device)
                
                output = self.model(x_batch, edge_index_batch, edge_type_batch)
                output = output[0: batch_size]

                all_outputs.append(output.cpu())
                all_labels.append(labels.cpu())
            
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            soft_labels = torch.softmax(all_outputs / self.temperature, dim=1)
            soft_labels[self.train_idx] = all_labels[self.train_idx]

            torch.save(soft_labels, self.intermediate_data_filepath / f'soft_labels_iter_{self.iter}.pt')

            self.iter += 1


    def eval(self, embeddings_LM,  mode='valid'):
        if mode == 'valid':            
            eval_loader =  build_GNN_dataloader(self.dataloader_config, self.valid_idx, embeddings_LM,  self.hard_labels, self.edge_index, self.edge_type, mode='eval')
        elif mode == 'test':
            eval_loader =  build_GNN_dataloader(self.dataloader_config, self.test_idx, embeddings_LM, self.hard_labels, self.edge_index, self.edge_type, mode='eval')
        self.model.eval()

        valid_predictions = []
        valid_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                batch_size = batch.batch_size
                x_batch = batch.x.to(self.device)
                edge_index_batch = batch.edge_index.to(self.device)
                edge_type_batch = batch.edge_type.to(self.device)
                labels = batch.labels[0: batch_size].to(self.device)
                
                output = self.model(x_batch, edge_index_batch, edge_type_batch)
                output = output[0: batch_size]

                valid_predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                valid_labels.append(torch.argmax(labels, dim=1).cpu().numpy())

            valid_predictions = np.concatenate(valid_predictions)
            valid_labels = np.concatenate(valid_labels)
            valid_acc = accuracy_score(valid_labels, valid_predictions)
            valid_f1 = f1_score(valid_labels, valid_predictions)

            return valid_acc, valid_f1

        

    def test(self, embeddings_LM):
        print('Computing test accuracy and f1 for GNN...')
        ckpt = torch.load(self.ckpt_filepath / 'best.pkl')
        self.model.load_state_dict(ckpt['model'])
        test_acc, test_f1 = self.eval(embeddings_LM, 'test')
        print(f'GNN Test Accuracy = {test_acc}')
        print(f'GNN Test F1 = {test_f1}')
        self.run.log({'GNN Test Accuracy': test_acc})
        self.run.log({'GNN Test F1': test_f1})
        self.results['accuracy'] = test_acc
        self.results['f1'] = test_f1

    def load_soft_labels(self, iter):
        soft_labels = torch.load(self.intermediate_data_filepath / f'soft_labels_iter_{iter}.pt')
        return soft_labels

    def save_results(self, path):
        json.dump(self.results, open(path, 'w'), indent=4)
        
    def get_train_idx_all(self):
        n_total = self.hard_labels.shape[0]
        all = set(np.arange(n_total))
        exclude = set(self.train_idx.numpy())
        n = self.train_idx.shape[0]
        pl_ratio_GNN = min(self.pl_ratio, (n_total - n) / n)
        n_pl_GNN = int(n * pl_ratio_GNN)
        self.pl_idx = torch.LongTensor(np.random.choice(np.array(list(all - exclude)), n_pl_GNN, replace=False))
        self.train_idx_all = torch.cat((self.train_idx, self.pl_idx))
        self.is_pl = torch.ones_like(self.train_idx_all, dtype=torch.int64)
        self.is_pl[0: n] = 0
        
        


class MLP_Trainer:
    def __init__(
        self, 
        device, 
        optimizer_name,
        lr,
        weight_decay,
        dropout,
        pl_weight,
        batch_size,
        n_layers,
        hidden_dim,
        activation,
        glnn_epochs,
        mlp_epochs_per_iter,
        temperature,
        pl_ratio,
        intermediate_data_filepath,
        ckpt_filepath,
        KD_ckpt_filepath,
        train_idx,
        valid_idx,
        test_idx,
        hard_labels,
        run,
        seed,
        use_gnn):

        self.device = device
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.pl_weight = pl_weight
        self.dropout = dropout        
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.mlp_epochs_per_iter = mlp_epochs_per_iter
        self.glnn_epochs = glnn_epochs
        self.temperature = temperature
        self.pl_ratio = pl_ratio
        self.intermediate_data_filepath = intermediate_data_filepath    
        self.ckpt_filepath = ckpt_filepath
        self.KD_ckpt_filepath = KD_ckpt_filepath
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.hard_labels = hard_labels
        self.run = run
        self.seed = seed
        self.use_gnn = use_gnn
        self.iter = 0
        self.best_iter = 0
        self.best_valid_acc = 0
        self.best_valid_epoch = 0
        self.criterion = CrossEntropyLoss()
        self.KD_criterion = KLDivLoss(log_target=False, reduction='batchmean')

        
        self.get_train_idx_all()
        self.results = {}

        self.dataloader_config = {
            'batch_size': batch_size
            }
        
        self.optimizer_args = dict(lr=lr, weight_decay=weight_decay)
        
    def get_scheduler(self, optimizer, T_max):
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
    

    def get_optimizer(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(self.model.parameters(), **self.optimizer_args)
        else:
            return NotImplementedError
        
        return optimizer

    def build_model(self):
        if self.use_gnn:
            self.model = MLP(in_channels=768, hidden_channels=self.hidden_dim, out_channels=2, dropout=self.dropout, act=self.activation, num_layers=self.n_layers).to(self.device)
        else:
            ckpt = torch.load(self.KD_ckpt_filepath / f'seed_{self.seed}_best.pkl')
            self.model = MLP(**ckpt['model_params']).to(self.device)
            # self.model.load_state_dict(ckpt['model'])

    
    def KD_GLNN(self, LM_embeddings, soft_labels):
        print('Distilling from GNN to GLNN')
        train_loader = build_MLP_dataloader(self.dataloader_config, self.train_idx_all, LM_embeddings, soft_labels, mode='train', is_pl=self.is_pl)
        
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer, self.glnn_epochs)
        
        for epoch in tqdm(range(self.glnn_epochs)):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                LM_embedding, label, is_pl = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                output = self.model(LM_embedding)

                pl_idx = torch.nonzero(is_pl == 1).squeeze()
                rl_idx = torch.nonzero(is_pl == 0).squeeze()

                if pl_idx.numel() == 0:
                    loss = self.criterion(output[rl_idx], label[rl_idx])
                elif rl_idx.numel() == 0:
                    loss = self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), label[pl_idx])
                else:
                    # loss = self.pl_weight * self.criterion(output[pl_idx], labels[pl_idx]) + (1 - self.pl_weight) * self.criterion(output[rl_idx], labels[rl_idx])
                    loss = self.pl_weight * self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), label[pl_idx]) + (1 - self.pl_weight) * self.criterion(output[rl_idx], label[rl_idx])

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.run.log({'GLNN KD Train Loss': loss.item()})

    
            valid_acc, valid_f1 = self.eval(LM_embeddings)
            
            self.run.log({'GLNN KD Valid Accuracy': valid_acc})
            self.run.log({'GLNN KD Valid F1': valid_f1})

            if valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
                self.best_epoch = epoch
                torch.save({'model': self.model.state_dict(), 'model_params': {'num_layers': self.n_layers, 'hidden_channels': self.hidden_dim, 'dropout': self.dropout, 'act': self.activation, 'in_channels': 768, 'out_channels': 2}}, self.KD_ckpt_filepath / f'seed_{self.seed}_best.pkl')

        print(f'The highest valid accuracy is {self.best_valid_acc}!')
        print(f'Save model from epoch {self.best_epoch}')
            
    def train(self, LM_embeddings, soft_labels):
        print('MLP training start!')
        print(f'This is iter {self.iter}')
        train_loader = build_MLP_dataloader(self.dataloader_config, self.train_idx_all, LM_embeddings, soft_labels, mode='train', is_pl=self.is_pl)
        early_stop_flag = True
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer, self.glnn_epochs)
        for epoch in tqdm(range(self.mlp_epochs_per_iter)):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                LM_embedding, label, is_pl = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                output = self.model(LM_embedding)

                pl_idx = torch.nonzero(is_pl == 1).squeeze()
                rl_idx = torch.nonzero(is_pl == 0).squeeze()

                if pl_idx.numel() == 0:
                    loss = self.criterion(output[rl_idx], label[rl_idx])
                elif rl_idx.numel() == 0:
                    loss = self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), label[pl_idx])
                else:
                    loss = self.pl_weight * self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), label[pl_idx]) + (1 - self.pl_weight) * self.criterion(output[rl_idx], label[rl_idx])

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.run.log({'MLP Train Loss': loss.item()})

            

            valid_acc, valid_f1 = self.eval(LM_embeddings)
            
            self.run.log({'MLP Valid Accuracy': valid_acc})
            self.run.log({'MLP Valid F1': valid_f1})

            if valid_acc > self.best_valid_acc:
                early_stop_flag = False
                self.best_valid_acc = valid_acc
                self.best_epoch = epoch
                self.best_iter = self.iter
                torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.ckpt_filepath / 'best.pkl')
        print(f'The highest valid accuracy is {self.best_valid_acc}!')
        return early_stop_flag

    def infer(self, LM_embeddings):
        self.model.eval()
        infer_loader = build_MLP_dataloader(self.dataloader_config, None, LM_embeddings,  self.hard_labels, mode='infer')
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            ckpt = torch.load(self.ckpt_filepath / 'best.pkl')
            self.model.load_state_dict(ckpt['model'])
            for batch in infer_loader:
                LM_embedding, label = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(LM_embedding)
                all_outputs.append(output.cpu())
                all_labels.append(label.cpu())
                
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            soft_labels = torch.softmax(all_outputs / self.temperature, dim=1)
            soft_labels[self.train_idx] = all_labels[self.train_idx]

            torch.save(soft_labels, self.intermediate_data_filepath / f'soft_labels_iter_{self.iter}.pt')
            
            self.iter += 1

    def eval(self, LM_embeddings, mode='valid'):
        if mode == 'valid':            
            eval_loader =  build_MLP_dataloader(self.dataloader_config, self.valid_idx, LM_embeddings, self.hard_labels, mode='eval')
        elif mode == 'test':
            eval_loader =  build_MLP_dataloader(self.dataloader_config, self.test_idx, LM_embeddings, self.hard_labels, mode='eval')
        self.model.eval()

        valid_predictions = []
        valid_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                LM_embedding, label = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(LM_embedding)

                valid_predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                valid_labels.append(torch.argmax(label, dim=1).cpu().numpy())

            valid_predictions = np.concatenate(valid_predictions)
            valid_labels = np.concatenate(valid_labels)
            valid_acc = accuracy_score(valid_labels, valid_predictions)
            valid_f1 = f1_score(valid_labels, valid_predictions)

            return valid_acc, valid_f1
        
    def test(self, LM_embeddings):
        print('Computing test accuracy and f1 for MLP...')
        ckpt = torch.load(self.ckpt_filepath / 'best.pkl')
        self.model.load_state_dict(ckpt['model'])
        test_acc, test_f1 = self.eval(LM_embeddings, 'test')
        print(f'MLP Test Accuracy = {test_acc}')
        print(f'MLP Test F1 = {test_f1}')
        self.run.log({'MLP Test Accuracy': test_acc})
        self.run.log({'MLP Test F1': test_f1})
        self.results['accuracy'] = test_acc
        self.results['f1'] = test_f1

    def save_results(self, path):
        json.dump(self.results, open(path, 'w'), indent=4)
    
    def get_train_idx_all(self):
        n_total = self.hard_labels.shape[0]
        all = set(np.arange(n_total))
        exclude = set(self.train_idx.numpy())
        n = self.train_idx.shape[0]
        pl_ratio_LM = min(self.pl_ratio, (n_total - n) / n)
        n_pl_LM = int(n * pl_ratio_LM)
        pl_idx = torch.LongTensor(np.random.choice(np.array(list(all - exclude)), n_pl_LM, replace=False))
        self.train_idx_all = torch.cat((self.train_idx, pl_idx))
        self.is_pl = torch.ones_like(self.train_idx_all, dtype=torch.int64)
        self.is_pl[0: n] = 0

