
import os
os.environ["OMP_NUM_THREADS"] = "1"
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from nvwa_downstream_pred import Nvwa_enchane_SimVP
from tqdm import tqdm
from API import *
from utils import *
import logging


class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self.get_data()
        self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu-1)
            # print(torch.ones(3, 3).cuda())
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self.get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = Nvwa_enchane_SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T, args=args).to(self.device)

    def get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _save(self, name=''):
        save_path = os.path.join(self.checkpoints_path, name + '.pth')
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'args': self.args.__dict__  # Save configuration
        }
        torch.save(checkpoint, save_path)
        # Debugging: Check if the file exists
        if os.path.exists(save_path):
            print_log(f"Checkpoint saved successfully at {save_path}")
        else:
            print_log(f"Failed to save checkpoint at {save_path}")
        
    def _load_checkpoint(self, checkpoint_name=''):
        checkpoint_path = os.path.join(self.checkpoints_path, checkpoint_name + '.pth')
        
        # First check if checkpoints directory exists
        if not os.path.exists(self.checkpoints_path):
            print_log(f"Checkpoints directory not found at {self.checkpoints_path}")
            return 0

        # Then check if specific checkpoint exists
        if not os.path.exists(checkpoint_path):
            print_log(f"No checkpoint found at {checkpoint_path}")
            
            # Try to find the latest checkpoint
            checkpoints = [f for f in os.listdir(self.checkpoints_path) if f.endswith('.pth')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(self.checkpoints_path, x)))
                checkpoint_path = os.path.join(self.checkpoints_path, latest_checkpoint)
                print_log(f"Loading latest checkpoint found: {checkpoint_path}")
            else:
                print_log("No checkpoints found. Starting from scratch.")
                return 0

        try:
            print_log(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            print_log(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
            return start_epoch
            
        except Exception as e:
            print_log(f"Error loading checkpoint: {str(e)}")
            return 0

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)

        # Load checkpoint if available
        start_epoch = self._load_checkpoint('checkpoint')  # Use 'checkpoint' or a specific epoch file
        print_log(f"Starting training from epoch {start_epoch}")

        max_iterations = 5  # Debug: limit iterations per epoch

        for epoch in range(start_epoch, config['epochs']):
            train_loss = []
            self.current_epoch = epoch  # Track the current epoch
            self.model.train()
            train_pbar = tqdm(self.train_loader, total=max_iterations)

            # for batch_x, batch_y in train_pbar:
            for i, (batch_x, batch_y) in enumerate(train_pbar):
                if i >= max_iterations:  # Exit loop after max_iterations
                    break

                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)
                loss = self.criterion(pred_y, batch_y)
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss = np.average(train_loss)
            print_log(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss}")

            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=f'checkpoint_epoch_{epoch}')  # Save epoch-specific checkpoint
                        self._save(name='checkpoint')  # Save the current state as 'checkpoint'
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, vali_loss))
                recorder(vali_loss, self.model, self.path)

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        

        max_val_iterations = 5

        vali_pbar = tqdm(vali_loader, total=max_val_iterations)

        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i >= max_val_iterations:  # Exit loop after max_val_iterations
                break

            # if i * batch_x.shape[0] > 1000:
            #     break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            print(batch_x.size())
            pred_y = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            loss = self.criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        # print(len(preds_lst))
        # print(len(preds_lst[0]))
        # print(len(preds_lst[0][0]))
        # print(len(preds_lst[0][0][0]))
        # print(len(preds_lst[0][0][0][0]))
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        # print(len(preds))
        # print(len(preds[0]))
        # print(len(preds[0][0]))
        # print(len(preds[0][0][0]))
        # print(len(preds[0][0][0][0]))
        # print(preds.size(),trues.size())
        # mse, mae = metric(preds, trues, vali_loader.dataset.dataset.mean, vali_loader.dataset.dataset.std, True)
        mse, mae = metric(preds, trues, self.data_mean, self.data_std, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}'.format(mse, mae))
        self.model.train()
        return total_loss

    def test(self, args):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in self.test_loader:
            pred_y = self.model(batch_x.to(self.device))
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # mse, mae, ssim, psnr = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, True)
        # print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        mse, mae = metric(preds, trues, self.test_loader.dataset.dataset.mean, self.test_loader.dataset.dataset.std, True)
        print_log('mse:{:.4f}, mae:{:.4f}'.format(mse, mae))

        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return mse