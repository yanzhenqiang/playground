import time
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import yload, ydump, mkdir, bmv, bmtm, bmtv, bmmt, SO3
from datetime import datetime

import networks
import dataset

import matplotlib.pyplot as plt
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
figsize = (20, 12)


class GyroLearningBasedProcessing:
    def __init__(self, configs):
        self.configs = configs

        self.net = networks.GyroNet(**self.configs['net_params'])
        # if test:
        #     datetime.now().strftime("%Y%m%d%H%M%S")
        #     self.configs = yload(self.address, 'config.yaml')
        #     self.load_weights()
        
        self.result = {}
        self.freq = 100

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        self.net.load_state_dict(weights)
        self.net.cuda()

    def save_weights(self, weights_path):
        self.net.eval().cpu()
        torch.save(self.net.state_dict(), weights_path)
        self.net.train().cuda()

    def train(self):
        ydump(self.configs, self.configs['result_dir'], 'configs.yaml')

        dataset_train = dataset.BaseDataset(self.configs, mode='train')
        dataset_val = dataset.BaseDataset(self.configs, mode='val')
        dataloader = DataLoader(dataset_train, **self.configs['dataloader'])

        # define optimizer, scheduler and loss
        Optimizer = self.configs['optimizer_class']
        Scheduler = self.configs['scheduler_class']
        Loss = self.configs['loss_class']

        optimizer_params = self.configs['optimizer']
        scheduler_params = self.configs['scheduler']
        loss_params = self.configs['loss']

        optimizer = Optimizer(self.net.parameters(), **optimizer_params)
        scheduler = Scheduler(optimizer, **scheduler_params)
        criterion = Loss(**loss_params)

        # init net
        # TODO(Zhenqiang):Too trick.
        self.net = self.net.cuda()
        mean_u, std_u = dataset_train.init_normalize_factors()
        self.net.set_normalized_factors(mean_u.cpu(), std_u.cpu())

        best_loss = torch.Tensor([float('Inf')])

        for epoch in range(1, self.configs['n_epochs'] + 1):
            loss_epoch = 0
            optimizer.zero_grad()
            for imus, xs, dp in dataloader:
                # add_noise
                imu_std = torch.Tensor([8e-5, 1e-3]).float().cuda()
                imu_b0 = torch.Tensor([1e-3, 1e-3]).float().cuda()
                noise = torch.randn_like(imus).cuda()
                noise[:, :, :3] = noise[:, :, :3] * imu_std[0]
                noise[:, :, 3:6] = noise[:, :, 3:6] * imu_std[1]
                b0 = torch.distributions.uniform.Uniform(-torch.ones(1), torch.ones(1)).sample(imus[:, 0].shape).cuda()
                b0[:, :, :3] = b0[:, :, :3] * imu_b0[0]
                b0[:, :, 3:6] =  b0[:, :, 3:6] * imu_b0[1]
                imus = imus + noise + b0.transpose(1, 2)
                
                hat_xs = self.net(imus)
                loss = criterion(xs.cuda(), dp.cuda(), hat_xs)/len(dataloader)
                loss.backward()
                loss_epoch += loss.detach().cpu()
            optimizer.step()

            print('Train Epoch: {:2d} \tLoss: {:.4f}'.format(
                epoch, loss_epoch.item()))

            scheduler.step(epoch)
            if epoch % self.configs['freq_val'] == 0:
                loss_epoch = 0
                self.net.eval()
                with torch.no_grad():
                    for i in range(len(dataset_val)):
                        imus, xs, dp = dataset_val[i]
                        hat_xs = self.net(imus.cuda().unsqueeze(0))
                        loss = criterion(xs.cuda().unsqueeze(0), dp.cuda().unsqueeze(0), hat_xs)/len(dataset_val)
                        loss_epoch += loss.cpu()
                self.net.train()

                if 0.5*loss_epoch <= best_loss:
                    best_loss = loss_epoch
                    self.save_weights(self.configs['weights_path'])

        dict_loss = {
            'final_loss/val': best_loss.item(),
            }

    def test(self):
        Loss = self.configs['loss_class']
        loss_params = self.configs['loss']
        criterion = Loss(**loss_params)

        dataset_test = dataset.BaseDataset(self.configs, mode='test')
        self.net.eval()
        for i in range(len(dataset_test)):
            seq = dataset_test.sequences[i]
            imus, xs, dp = dataset_test[i]
            with torch.no_grad():
                hat_xs = self.net(imus.cuda().unsqueeze(0))
            loss = criterion(xs.cuda().unsqueeze(0), dp.cuda().unsqueeze(0), hat_xs)
            mkdir(self.configs['result_dir'], seq)
            self.result[seq] = {
                'hat_xs': hat_xs[0],
                'loss': loss.cpu().item(),
            }

        self.display(dataset_test)

    def display(self, ds):
        for i, seq in enumerate(ds.sequences):
            print('\n', 'Results for sequence ' + seq )
            self.seq = seq
            # get ground truth
            keys = sorted(ds.sequence_data.keys())
            self.gt = ds.sequence_data[keys[i]]
            Rots = SO3.from_quaternion(self.gt['q'])
            self.gt['Rots'] = Rots
            self.gt['rpys'] = SO3.to_rpy(Rots)
            # get data and estimate
            self.net_us = self.result[seq]['hat_xs']
            self.raw_us, _, _ = ds[i]
            yaw, pitch, roll = -self.raw_us[:, 0], self.raw_us[:, 1], self.raw_us[:, 2]
            self.raw_us =torch.stack((pitch, roll, yaw), dim = 1)

            N = self.net_us.shape[0]
            l = 180/np.pi
            self.gyro_corrections =  (self.raw_us[:, :3] - self.net_us[:N, :3])*l
            self.gt['rpys'] *= l
            self.ts = torch.linspace(0, N*self.configs['loss']['dt'], N)
            self.plot()

    def integrate_with_quaternions(self, N, raw_us, net_us):
        imu_qs = SO3.qnorm(SO3.qexp(raw_us[:, :3].cuda().double()*self.configs['loss']['dt']))
        net_qs = SO3.qnorm(SO3.qexp(net_us[:, :3].cuda().double()*self.configs['loss']['dt']))
        Rot0 = SO3.qnorm(self.gt['q'][:2].cuda().double())
        imu_qs[0] = Rot0[0]
        net_qs[0] = Rot0[0]

        N = np.log2(imu_qs.shape[0])
        for i in range(int(N)):
            k = 2**i
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:-k], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:-k], net_qs[k:]))

        if int(N) < N:
            k = 2**int(N)
            k2 = imu_qs[k:].shape[0]
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:k2], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:k2], net_qs[k:]))

        imu_Rots = SO3.from_quaternion(imu_qs).float()
        net_Rots = SO3.from_quaternion(net_qs).float()
        return net_qs.cpu(), imu_Rots, net_Rots

    def plot(self):
        N = self.raw_us.shape[0]
        raw_us = self.raw_us[:, :3]
        net_us = self.net_us[:, :3]

        net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions(N, raw_us, net_us)
        imu_rpys = 180/np.pi*SO3.to_rpy(imu_Rots).cpu()
        net_rpys = 180/np.pi*SO3.to_rpy(net_Rots).cpu()
        self.plot_orientation(imu_rpys, net_rpys, N)
        self.plot_orientation_error(imu_Rots, net_Rots, N)

    def plot_orientation(self, imu_rpys, net_rpys, N):
        title = "Orientation estimation"
        gt = self.gt['rpys'][:N]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(self.ts, gt[:, i].cpu(), color='black', label=r'ground truth')
            axs[i].plot(self.ts, imu_rpys[:, i].cpu(), color='red', label=r'raw IMU')
            axs[i].plot(self.ts, net_rpys[:, i].cpu(), color='blue', label=r'net IMU')
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        self.savefig(axs, fig, 'orientation')

    def plot_orientation_error(self, imu_Rots, net_Rots, N):
        gt = self.gt['Rots'][:N].cuda()
        raw_err = 180/np.pi*SO3.log(bmtm(imu_Rots, gt)).cpu()
        net_err = 180/np.pi*SO3.log(bmtm(net_Rots, gt)).cpu()
        title = "$SO(3)$ orientation error"
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(self.ts, raw_err[:, i], color='red', label=r'raw IMU')
            axs[i].plot(self.ts, net_err[:, i], color='blue', label=r'net IMU')
            axs[i].set_ylim(-10, 10)
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        self.savefig(axs, fig, 'orientation_error')

    def savefig(self, axs, fig, name):
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
                axs[i].legend()
        else:
            axs.grid()
            axs.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.configs['result_dir'], self.seq, name + '.png'))

