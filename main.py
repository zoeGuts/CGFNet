import os
import argparse
import random
import numpy as np
import torch

from solver import Solver

parser = argparse.ArgumentParser(description='Aerial Scene Classification.')

parser.add_argument('-d', '--device_ids', type=int, default=1, metavar='GPU', help="number of gpus to use")
parser.add_argument('--method', type=str, default='ours', 
                    help='method (options:Baseline, CBAM, MBLANet, ours)')
parser.add_argument('--arch', type=str, default='resnet50', help='major encoder')
parser.add_argument('--plot_acc_curv', type=bool, default=False, help='Draw a graph of accuracy')

# Method setting
parser.add_argument('-v', '--method_version', type=str, default='original', help='method version')

parser.add_argument('--no_spatial', default=False, action='store_true', help='channel attention only!')
parser.add_argument('-g', '--channel_groups', type=int, default=64, help='channel_groups')
parser.add_argument('-k', '--sa_kernel_size', type=int, default=3)

parser.add_argument('--no_lofct', default=False, action='store_true', help='cross entrpy only!')
parser.add_argument('--beta', type=float, default=4.0, help='beta')

parser.add_argument('-r', '--reduction_ratio', type=int, default=16, help='reduction_ratio')

# Dataset settings
parser.add_argument('--dataset', type=str, default='AID02', 
                    help='dataset (options: ERA, UCM05, UCM08, AID02, AID05) (default: AID02)')
parser.add_argument('--train_dir', type=str, default=None, help='training set path (default: None)')
parser.add_argument('--val_dir', type=str, default=None, help='validation set path (default: None)')
parser.add_argument('-c', '--numberofclass', type=int, default=1000, help='number of categories') 
parser.add_argument('-augmentation', type=str, default=None, 
                    help='the type of data augmentation (options: ) (default: )')

# Training settings
parser.add_argument('-b', '--batch_size', type=int, default=32, help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', type=int, default=100, 
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start_epoch', type=int, default=0, help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--lr_type', default='cosine', type=str, choices=['cosine', 'multistep', 'step30'],
                    help='learning rate strategy (default: cosine)')
parser.add_argument('--lrf', type=float, default=0.01)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, 
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1234, help='random seed')

parser.add_argument('--load_model', type=str, default=None, help='path to load model (default:None)')
parser.add_argument('--freeze_layers', default=False, action='store_true', help='freeze layers') 
parser.add_argument('--no_pretrained', default=True, dest='pretrained', action='store_false', 
                    help='to load pre-trained weights')
parser.add_argument('--pretrained_weights', type=str, default=None, help='path to load pre-trained weights')
parser.add_argument('--resume', type=str, default=None, 
                    help='path to latest checkpoint (default: None)') 

parser.add_argument('--nw', type=int, default=4, help='number of data loading workers (default: 4)')
parser.add_argument('--pin_memory', type=bool, default=False)
parser.add_argument('--print_freq', '-p', default=10, type=int, 
                    help='print frequency (default: 10)')

parser.add_argument('--evaluate', default=False, dest='evaluate', action='store_true', 
                    help='evaluation only')
parser.add_argument('--save_dir', type=str, default=None, help='dir to save model')
parser.add_argument('--expname', type=str, default=None, help='name of experiment')


def main():
    args = parser.parse_args()
    assert args.method in ['Baseline', 'CBAM', 'MBLANet', 'ours'], "method must be the one of ['Baseline', 'CBAM', 'MBLANet', 'ours',]. "
        
    if args.seed is not None:
        print('set the same seed for all.....')
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # set the dataset path
    data_root = r'./datasets/'
    if args.dataset == 'ERA':
        args.train_dir = os.path.join(data_root, 'ERA/Tra/')
        args.val_dir = os.path.join(data_root, 'ERA/Test/') 
        args.numberofclass = 25
    elif args.dataset == 'AID02':
        args.train_dir = os.path.join(data_root, 'AID_02/Tra/')
        args.val_dir = os.path.join(data_root, 'AID_02/Test/') 
        args.numberofclass = 30
    elif args.dataset == 'AID05':
        args.train_dir = os.path.join(data_root, 'AID_05/Tra/')
        args.val_dir = os.path.join(data_root, 'AID_05/Test/')
        args.numberofclass = 30
    elif args.dataset == 'UCM05':
        args.train_dir = os.path.join(data_root, 'UCMerced_LandUse_05/Tra/')
        args.val_dir = os.path.join(data_root, 'UCMerced_LandUse_05/Test/')
        args.numberofclass = 21
    elif args.dataset == 'UCM08':
        args.train_dir = os.path.join(data_root, 'UCMerced_LandUse_08/Tra/')
        args.val_dir = os.path.join(data_root, 'UCMerced_LandUse_08/Test/')
        args.numberofclass = 21
    elif args.dataset == 'OPTIMAL31_08':
        args.train_dir = os.path.join(data_root, 'OPTIMAL31_08/Tra/')
        args.val_dir = os.path.join(data_root, 'OPTIMAL31_08/Test/')
        args.numberofclass = 31
    elif args.dataset == 'NWPU45_02':
        args.train_dir = os.path.join(data_root, 'NWPU45_02/Tra/')
        args.val_dir = os.path.join(data_root, 'NWPU45_02/Test/')
        args.numberofclass = 45
    elif args.dataset == 'NWPU45_01':
        args.train_dir = os.path.join(data_root, 'NWPU45_01/Tra/')
        args.val_dir = os.path.join(data_root, 'NWPU45_01/Test/')
        args.numberofclass = 45
    elif args.dataset == 'RSSCN7_05':
        args.train_dir = os.path.join(data_root, 'RSSCN7_05/Tra/')
        args.val_dir = os.path.join(data_root, 'RSSCN7_05/Test/')
        args.numberofclass = 7
        
    # pre-training weights
    if args.pretrained:
        if args.arch == 'resnet50':
            args.pretrained_weights = './path/resnet50-19c8e357.pth'
        
    if args.expname is None:
        args.expname = '%s_%s_%s' % (args.method, args.arch, args.dataset)  # example: cbam_resnet_era
    if args.save_dir is None:
        args.save_dir = f'./runs/{args.expname}'
        os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./fig'):
        os.mkdir('./fig')
    
    args.nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(args.nw))
    
    solver = Solver(args)
    
    if args.evaluate is False:
        solver.train()
    else:
        pass
   
if __name__ == '__main__':
   main()
    