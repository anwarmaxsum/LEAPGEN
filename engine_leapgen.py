# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
import time
from typing import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.nn import MSELoss

from timm.utils import accuracy
from timm.optim import create_optimizer
import copy
import utils
from torch.distributions.multivariate_normal import MultivariateNormal
import logging

# for attribute matching of tasks
from attribute_matching import num_new_prompts, get_descriptors_embedding1
from timm.scheduler import create_scheduler

# import clip
import random

# from kan import *

def train_one_epoch(model: torch.nn.Module, 
                    criterion, data_loader: Iterable, 
                    device: torch.device, epoch: int, max_norm: float = 0,
                    optimizer=None,
                    old_prompt_matcher = None,
                    old_prompt = None,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,
                    old_num_k=5,):

    model.train(set_training_mode)

    s = old_num_k

    # Freezing previous tasks' filters
    for name, param in model.named_parameters():
        if name.find('e_prompt.v_conv_vals') >=0  or name.find('e_prompt.k_conv_vals') >=0:
            for i in range(s):
                if name.find('.{}.weight'.format(i)) >=0 or name.find('.{}.bias'.format(i)) >=0:
                    param.requires_grad = False


    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    
    # metric_logger.add_meter('Lr_head', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    if args.SLCA:
        metric_logger.add_meter('Lr_cls', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Lr_rps', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input, task_id=task_id, train=set_training_mode)
        logits = output['logits']

        # Masking and computing loss
        known_classes = task_id*len(class_mask[0])
        cur_targets = torch.where(target-known_classes>=0,target-known_classes,-100)
        loss = criterion(logits[:, known_classes:], cur_targets) # base criterion (CrossEntropyLoss)


        if args.use_e_prompt or args.use_g_prompt:
            if task_id > 0:
                l1_loss = 0.0
                for old_wt, new_wt in zip(old_prompt_matcher.parameters(), model.e_prompt.prompt_embed_matcher.parameters()):
                    l1_loss += torch.norm(old_wt.detach() - new_wt, p=1)
                loss = loss + 0.01 * l1_loss
                prompt_loss = torch.norm(old_prompt.detach() - model.e_prompt.prompt, p=1)
                loss = loss + 0.01 * prompt_loss


        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        if args.use_clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        
        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        if args.SLCA:
            metric_logger.update(Lr_cls=optimizer.param_groups[0]["lr"])
            metric_logger.update(Lr_rps=optimizer.param_groups[1]["lr"]) 
        else:
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    logging.info("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_with_aux(model: torch.nn.Module,original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable,
                    aux_desc_emb, 
                    device: torch.device, epoch: int, max_norm: float = 0,
                    optimizer=None,
                    old_prompt_matcher = None,
                    old_prompt = None,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,
                    old_num_k=5,):

    model.train(set_training_mode)

    s = old_num_k

    # Freezing previous tasks' filters
    for name, param in model.named_parameters():
        if name.find('e_prompt.v_conv_vals') >=0  or name.find('e_prompt.k_conv_vals') >=0:
            for i in range(s):
                if name.find('.{}.weight'.format(i)) >=0 or name.find('.{}.bias'.format(i)) >=0:
                    param.requires_grad = False


    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    

    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    

    if args.dualopt:
        k_params = []
        for n, p in model.named_parameters():
            if n.find('prompt_key2') >=0:
                k_params.append(p)
        k_network_params = [{'params': k_params, 'lr': args.lr*args.k_mul, 'weight_decay': args.weight_decay}]
        task_optimizer = optim.Adam(k_network_params, weight_decay=args.weight_decay)


    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))

    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    old_key_l = copy.deepcopy(model.e_prompt.prompt_key)
    old_key2 = copy.deepcopy(model.e_prompt.prompt_key2)

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
                
        output = model.forwardA1(input,target, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']
        
        loss = args.intertask_coeff*criterion(logits, target)

        
        # Masking and computing loss
        known_classes = task_id*len(class_mask[0])
        cur_targets = torch.where(target-known_classes>=0,target-known_classes,-100)
        loss += criterion(logits[:, known_classes:], cur_targets) # base criterion (CrossEntropyLoss)
        

        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        loss2 = 0
        if args.pull_constraint and 'reduce_sim2' in output:
            if args.dualopt:
                loss2 = -1 * args.pull_constraint_coeff2 * output['reduce_sim2']
            else:
                loss = loss - args.pull_constraint_coeff2 * output['reduce_sim2']
            # print("Similarity : ", output['reduce_sim'].item(), " Similarity 2 : ", output['reduce_sim2'].item())

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        if args.use_clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        if args.dualopt:
            task_optimizer.zero_grad()
            loss2.backward() 
            task_optimizer.step()


        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    logging.info("Averaged stats: {}".format(metric_logger))

    model.eval()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader,  
            device, task_id=-1, class_mask=None, args=None,):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    correct = 0
    total = 0

    # Batchwise Eval time
    start_eval_time = time.time()
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            output = model(input, task_id=task_id)
            logits = output['logits']
            

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)
            # print('Loss')
            predicts = torch.max(logits, dim=1)[1]
            correct += (predicts == target).sum()
            total += len(target)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    end_eval_time = time.time()
    # gather the stats from all processesde
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    logging.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    
    print(f"Batchwise eval time for task {task_id+1} = {(end_eval_time - start_eval_time)/len(data_loader)}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, total, correct


@torch.no_grad()
def evaluate_with_aux(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, aux_desc_emb,
            device, task_id=-1, class_mask=None, args=None,):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    correct = 0
    total = 0

    true_labels=[]
    pred_labels=[]

    # Batchwise Eval time
    start_eval_time = time.time()
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            
            # aux = get_nearest_embedding1(input, target, aux_desc_emb)

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None
                

            output = model.forwardA1(input,None, -1, cls_features=cls_features, train=False)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            max_label = torch.max(target).item() 
            if logits.shape[1] <= max_label:
                pf = torch.zeros((logits.shape[0],max_label+1-logits.shape[1]),device=logits.device)
                logits = torch.cat((logits,pf),dim=1)



            loss = criterion(logits, target)
            # print('Loss')
            predicts = torch.max(logits, dim=1)[1]
            correct += (predicts == target).sum()
            total += len(target)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            pred = torch.argmax(logits, dim=1)

            pred_labels.extend(pred.tolist())
            true_labels.extend(target.tolist())

    end_eval_time = time.time()
    # gather the stats from all processesde
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    logging.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    
    print(f"Batchwise eval time for task {task_id+1} = {(end_eval_time - start_eval_time)/len(data_loader)}")



    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, total, correct



@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    total, correct = 0, 0
    for i in range(task_id+1):
        logging.info('Evaluating task {}...'.format(i+1))
        test_stats, temp_total, temp_correct = evaluate(model=model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args,)

        # test_stats, temp_total, temp_correct = evaluate(model=model, data_loader=data_loader[i]['val'], 
        #                     device=device, task_id=i, class_mask=class_mask, args=args,)

        total += temp_total
        correct += temp_correct
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    final_acc = np.divide(correct.cpu(), total)*100.0
    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, final_acc, avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)
    logging.info(result_str)

    return test_stats


@torch.no_grad()
def evaluate_till_now_with_aux(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,aux_desc_emb, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    total, correct = 0, 0
    for i in range(task_id+1):
        logging.info('Evaluating task {}...'.format(i+1))

        test_stats, temp_total, temp_correct = evaluate_with_aux(model=model, original_model=original_model, data_loader=data_loader[i]['val'], aux_desc_emb=aux_desc_emb,
                            device=device, task_id=i, class_mask=class_mask, args=args,)

        total += temp_total
        correct += temp_correct
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    final_acc = np.divide(correct.cpu(), total)*100.0
    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, final_acc, avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)
    logging.info(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, lr_scheduler, optimizer, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    old_num_k = 0

    aux_desc_emb = None

    for task_id in range(args.num_tasks):

        print("Task number: ", task_id)
        if task_id == 0:
            aux_desc_emb = torch.stack(get_descriptors_embedding1(class_mask,task_id,args)).to(device)
        else:
            new_desc = torch.stack(get_descriptors_embedding1(class_mask,task_id,args)).to(device)
            aux_desc_emb = torch.cat((aux_desc_emb,new_desc),dim=0)
        print(aux_desc_emb.shape)

        if task_id>0:
            model.head.update(len(class_mask[task_id]))
            # model.add_new_head()
        print(model.head)
        
        not_n_params = []
        n_params = []
        k_params = []

        lrate_decay = 0.1
        param_list = list(model.parameters())
        if task_id:
            if args.dualopt:
                for n, p in model.named_parameters():
                    if n.find('prompt_key2') >=0:
                        pass
                    elif n.find('norm1')>=0 or n.find('norm2') >= 0 or n.startswith('norm') or n.find('fc_norm') >= 0:
                        n_params.append(p)
                    else:
                        not_n_params.append(p)
               
            else:
                for n, p in model.named_parameters():
                    if n.find('norm1')>=0 or n.find('norm2') >= 0 or n.startswith('norm') or n.find('fc_norm') >= 0:
                        n_params.append(p)
                    else:
                        not_n_params.append(p)
            
            network_params = [{'params': not_n_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
                             {'params': n_params, 'lr': 0.005*args.lr, 'weight_decay': args.weight_decay}] 
            # network_params = [{'params': not_n_params, 'lr': args.lr, 'weight_decay': args.weight_decay}] 
        else:
            if args.dualopt:
                for n, p in model.named_parameters():
                    if n.find('prompt_key2') >=0:
                        pass
                    else:
                        not_n_params.append(p)
                network_params = [{'params': not_n_params, 'lr': args.lr, 'weight_decay': args.weight_decay}]
            else:
                network_params = [{'params': param_list, 'lr': args.lr, 'weight_decay': args.weight_decay}]


    
        # if not args.SLCA:
        print("Using adam optimizer")
        print("Reinitialising optimizer")
        optimizer = optim.Adam(network_params, weight_decay=args.weight_decay)
        if args.sched != 'constant':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0000)

        elif args.sched == 'constant':
            lr_scheduler = None


        if args.use_e_prompt or args.use_g_prompt:
            curr_num_k = args.top_k_l 
            model.e_prompt.process_new_task(old_num_k, old_num_k + curr_num_k, aux_desc_emb)
        else:
            curr_num_k = 0
        
        old_prompt_matcher = None
        old_prompt = None
        
        if task_id > 5:
            tr_epochs = args.epochs+5
        else:
            tr_epochs = args.epochs


        for epoch in range(args.epochs):
        # for epoch in range(tr_epochs):     
            logging.info('Training for task {} epoch {}/{}'.format(task_id, epoch, args.epochs))     
            train_stats = train_one_epoch_with_aux(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'],
                                        aux_desc_emb=aux_desc_emb,
                                        optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad,
                                        old_prompt_matcher=old_prompt_matcher,
                                        old_prompt=old_prompt,
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,
                                        old_num_k=old_num_k,)
            # evaluate_with_aux(model=model, data_loader=data_loader[task_id]['train'], aux_desc_emb=aux_desc_emb,
            #                 device=device, task_id=task_id, class_mask=class_mask, args=args,)
            
            if lr_scheduler:
                lr_scheduler.step()


        if args.use_e_prompt or args.use_g_prompt:
            old_num_k += curr_num_k
        
        # if task_id >= 18:
        eval_start_time = time.time()
        test_stats = evaluate_till_now_with_aux(model=model, original_model=original_model, data_loader=data_loader, aux_desc_emb=aux_desc_emb, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        eval_end_time = time.time()
        print(f"Eval time for task {task_id+1} = {eval_end_time - eval_start_time}")

        
        if args.output_dir and utils.is_main_process():
            path = args.output_dir+'_'+args.dataset
            Path(os.path.join(path, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            
            checkpoint_path = os.path.join(path, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')




def classify_with_proto(data, proto):
    # print("Testing with prototype-based cosine cosine similarity")
    logits = []
    scale_factor = 10
  
    for i in range(0,len(proto)):
        class_proto = proto[i].cuda()
        c_logit = []
        for j in range(class_proto.shape[0]):
            logit = scale_factor * torch.nn.functional.cosine_similarity(
                    data,
                    class_proto[j],
                    dim=-1)
            c_logit.append(logit)
        # c_logit = torch.stack(c_logit).min(dim=0).values
        c_logit = torch.stack(c_logit).mean(dim=0)
        # print("c_logit shape")
        # print(c_logit.shape)

        # print(logit.shape)
        
        logits.append(c_logit)


    logits = torch.stack(logits).transpose(0,1)

    return logits


def get_nearest_embedding(data, label, proto):
    # print("Testing with prototype-based cosine cosine similarity")
    embeds = []
    scale_factor = 10
  
    # for i in range(0,len(proto)):
    # batch_proto = proto[label.tolist()]
    batch_proto = [proto[i] for i in label.tolist()]
    for i in range(len(batch_proto)):
        mylist = [*range(0, batch_proto[i].shape[0], 1)]
        idx = random.sample(mylist, 1)
        embeds.append(batch_proto[i][idx,:])

    embeds = torch.stack(embeds).cuda()
    # print(embeds.shape)
    return embeds


def get_nearest_embedding1(data, label, proto):
    # print("Testing with prototype-based cosine cosine similarity")
    embeds = []
    scale_factor = 10
  
    # for i in range(0,len(proto)):
    # batch_proto = proto[label.tolist()]
    batch_proto = [proto[i] for i in label.tolist()]
    # for i in range(len(batch_proto)):
    #     mylist = [*range(0, batch_proto[i].shape[0], 1)]
    #     idx = random.sample(mylist, 1)
    #     embeds.append(batch_proto[i][idx,:])

    embeds = torch.stack(batch_proto).cuda()
    embeds = embeds.unsqueeze(1)
    # print(embeds.shape)
    return embeds

