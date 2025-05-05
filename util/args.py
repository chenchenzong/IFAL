#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # basic arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--custom_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_dir', type=str, default='.', help='when to start saving models')    
    
    # federated learning arguments
    parser.add_argument('--rounds', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--test_bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--weight_decay', type=float, default=0.00001, help="weight decay (default: 0.00001)")
    parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay ratio")
    parser.add_argument('--reset', type=str, default='random_init', help='doing FL with queried dataset or not')
    parser.add_argument('--fl_algo', type=str, default='fedavg', help='federated learning algorithm')
    parser.add_argument('--mu', type=float, default=0.01, help='weight of FedProx regularization term')
    
    # dataset arguments
    parser.add_argument('--data_dir', type=str, default='./data', help='data path')
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--partition', type=str, default="dir_balance", help="methods for Non-IID")
    parser.add_argument('--dd_beta', type=float, default=0.1, help="beta for dirichlet distribution")
    parser.add_argument('--num_classes_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--imb_ratio', type=float, default=1.0, help="imbalance ratio for long tail dataset")
    
    # model arguments
    parser.add_argument('--model', type=str, default='cnn4conv', help='model name')
    
    # active learning arguments
    parser.add_argument('--resume_ratio', type=float, default=0., help="ratio of data examples for resume")
    parser.add_argument('--query_ratio', type=float, default=0.05, help="ratio of data examples per one query")
    parser.add_argument('--end_ratio', type=float, default=0.0, help="ratio for stopping query")
    parser.add_argument('--query_model_mode', type=str, default="global")
    parser.add_argument('--al_method', type=str, default=None)


    args = parser.parse_args()
    
    # popular benchmark
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.in_channels = 3
        args.img_size = 32
        if not args.end_ratio: args.end_ratio = 0.35
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.in_channels = 3
        args.img_size = 32
        if not args.end_ratio: args.end_ratio = 0.35
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
        args.in_channels = 3
        args.img_size = 64
        if not args.end_ratio: args.end_ratio = 0.2
    
    # for init
    if not args.resume_ratio:
        args.current_ratio = args.query_ratio
    else:
        args.current_ratio = args.resume_ratio
        
    args.data_dir += '/{}/'.format(args.dataset)
        
    return args