import argparse
from train_prediction import Exp
# from metrics import measurement_metrics
import os
import torch
import logging
from tqdm import tqdm
import numpy as np
from ConvHawkes.models.convhawkes import ConvHawkes
from nvwa_downstream_pred import Nvwa_enchane_SimVP
from utils import *
from API import *
from train_nuwa_hawkes import HawkesTrainer
import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    
    # Add a mode argument to specify which model to train
    parser.add_argument('--mode', type=str, required=True, choices=['nuwa', 'hawkes', 'both'],
                        help='Training mode: nuwa, hawkes, or both')
    
    # Base parameters
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--res_dir', default='./output/simvp_nighttime_mask', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--log_step', type=int, default=1, help='Frequency of logging and saving checkpoints (in epochs)')  # Add log_step here

    # Dataset parameters
    data_group = parser.add_argument_group('Dataset parameters')
    data_group.add_argument('--batch_size', default=128, type=int, help='Training batch size')
    data_group.add_argument('--val_batch_size', default=128, type=int, help='Validation/Test batch size')
    data_group.add_argument('--data_root', default='/home/ansingh/Nuwa_Hawkes/data/nighttime/', type=str)
    data_group.add_argument('--dataname', default='nighttime')
    data_group.add_argument('--num_workers', default=0, type=int)
    data_group.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process')
    data_group.add_argument('--max_patches', type=int, default=None, help='Maximum number of patches to process per file')


    # NuwaDynamics parameters (used when mode is 'nuwa' or 'both')
    nuwa_group = parser.add_argument_group('NuwaDynamics parameters')
    nuwa_group.add_argument('--in_shape', default=[2, 1, 32, 32], type=int, nargs='*')
    nuwa_group.add_argument('--hid_S', default=64, type=int)
    nuwa_group.add_argument('--hid_T', default=128, type=int)
    nuwa_group.add_argument('--N_S', default=4, type=int)
    nuwa_group.add_argument('--N_T', default=8, type=int)
    nuwa_group.add_argument('--groups', default=1, type=int)
    nuwa_group.add_argument('--epochs', default=1, type=int)
    nuwa_group.add_argument('--lr', default=0.00001, type=float)

    # Hawkes parameters (used when mode is 'hawkes' or 'both')
    hawkes_group = parser.add_argument_group('Hawkes parameters')
    hawkes_group.add_argument('--nuwa_checkpoint', type=str,
                             help='Path to pre-trained NuwaDynamics checkpoint (required for hawkes mode)')
    hawkes_group.add_argument('--hawkes_epochs', type=int, default=1)
    hawkes_group.add_argument('--hawkes_lr', type=float, default=0.001)
    hawkes_group.add_argument('--event_threshold', type=float, default=0.5)
    hawkes_group.add_argument('--beta', type=float, default=1.0)
    hawkes_group.add_argument('--sigma_k_scale', type=float, default=1.0)
    hawkes_group.add_argument('--sigma_zeta_scale', type=float, default=1.0)
    hawkes_group.add_argument('--mu', type=float, default=0.1)

    return parser

if __name__ == '__main__':
    args = create_parser().parse_args()
    
    # Validate arguments based on mode
    if args.mode in ['hawkes', 'both'] and args.nuwa_checkpoint is None:
        raise ValueError("--nuwa_checkpoint is required for hawkes or both mode")

    if args.mode == 'nuwa':
        # Train only NuwaDynamics
        nuwa_exp = Exp(args)
        print('>>>>>>>>>>>>>>>>>>>>>>>>> Training NuwaDynamics <<<<<<<<<<<<<<<<<<<<<<<<<')
        nuwa_model = nuwa_exp.train(args)
        print('>>>>>>>>>>>>>>>>>>>>>>>>> Testing NuwaDynamics <<<<<<<<<<<<<<<<<<<<<<<<<')
        nuwa_exp.test(args)

    elif args.mode == 'hawkes':
        # Train only Hawkes using pre-trained NuwaDynamics
        hawkes_trainer = HawkesTrainer(args)
        print('>>>>>>>>>>>>>>>>>>>>>>>>> Training Hawkes Model <<<<<<<<<<<<<<<<<<<<<<<<<')
        hawkes_trainer.train_hawkes()
        print('>>>>>>>>>>>>>>>>>>>>>>>>> Testing Hawkes Model <<<<<<<<<<<<<<<<<<<<<<<<<')
        # hawkes_trainer.validate_hawkes()

    else:  # mode == 'both'
        # Train NuwaDynamics first
        print('>>>>>>>>>>>>>>>>>>>>>>>>> Training NuwaDynamics <<<<<<<<<<<<<<<<<<<<<<<<<')
        nuwa_exp = Exp(args)
        nuwa_model = nuwa_exp.train(args)
        nuwa_exp.test(args)

        # Then train Hawkes using the newly trained NuwaDynamics
        print('>>>>>>>>>>>>>>>>>>>>>>>>> Training Hawkes Model <<<<<<<<<<<<<<<<<<<<<<<<<')
        args.nuwa_checkpoint = f"{args.res_dir}/{args.ex_name}/checkpoints/checkpoint.pth"
        hawkes_trainer = HawkesTrainer(args)
        hawkes_trainer.train()
        print('>>>>>>>>>>>>>>>>>>>>>>>>> Testing Hawkes Model <<<<<<<<<<<<<<<<<<<<<<<<<')
        hawkes_trainer.test()