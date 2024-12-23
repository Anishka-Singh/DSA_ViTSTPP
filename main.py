import argparse
from train_prediction import Exp
# from metrics import measurement_metrics
import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./output/simvp_taxibj_mask', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/TaxiBJ/', type=str)
    parser.add_argument('--dataname', default='taxibj')
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[4, 2, 32, 32], type=int, nargs='*')  # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj
    parser.add_argument('--hid_S', default=32, type=int)
    parser.add_argument('--hid_T', default=64, type=int)
    # parser.add_argument('--hid_S', default=64, type=int)
    # parser.add_argument('--hid_T', default=256, type=int)
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    
    return parser

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)

    load=exp.train_loader
    print(load)
    a=next(iter(load))
    print(a[0].shape)
    print(a[0].sum())
    print('>>>>>>>>>>>>>>>>>>>>>>>>> start <<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(args)
