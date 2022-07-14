import numpy as np
import os
import sys
import torch
from torch.utils.data.dataset import Dataset

from utils import SO3,bmtm

class BaseDataset(Dataset):
        
    def __init__(self, configs, mode):
        super().__init__()
        self.configs = configs

        self._train = False
        self._val = False
        self.mode = mode
        if mode == 'train':
            self._train = True
            self._val = False
        if mode =='val':
            self._train = False
            self._val = True

        self.min_train_freq = self.configs['min_train_freq']
        self.max_train_freq = self.configs['max_train_freq']

        self.sequences = self.configs[mode+'_seqs']
        self.sequence_data = {}
        self.read_data(self.configs['data_dir'])

    def __getitem__(self, i):
        keys = sorted(self.sequence_data.keys())
        mondict = self.sequence_data[keys[i]]
        N_max = mondict['xs'].shape[0]
        if self._train:
            n0 = torch.randint(0, self.max_train_freq, (1, ))
            nend = n0 + self.configs['N']
        elif self._val:
            n0 = self.max_train_freq + self.configs['N']
            nend = N_max - ((N_max - n0) % self.max_train_freq)
        else:
            n0 = 0
            nend = N_max - (N_max % self.max_train_freq)
        imu = mondict['imu'][n0: nend]
        xs = mondict['xs'][n0: nend]
        dp = mondict['dp'][n0: nend]
        return imu, xs, dp

    def __len__(self):
        return len(self.sequences)

    def length(self):
        return self._length

    def init_normalize_factors(self):
        print('Computing normalizing factors ...')
        num_data = 0
        mean_u = 0
        std_u = 0

        for sequence in self.sequences:
            data = self.sequence_data[sequence]
            us = data['imu']
            mean_u += us.sum(dim=0)
            num_data += us.shape[0]
        mean_u = mean_u / num_data

        for sequence in self.sequences:
            data = self.sequence_data[sequence]
            us = data['imu']
            std_u += ((us - mean_u) ** 2).sum(dim=0)
        std_u = (std_u / num_data).sqrt()
        
        print('mean_u    :', mean_u)
        print('std_u     :', std_u)
        return mean_u, std_u

    @staticmethod
    def interpolate(x, t, t_int):
        x_int = np.zeros((t_int.shape[0], x.shape[1]))
        for i in range(x.shape[1]):
            if i in [4, 5, 6, 7]:
                continue
            x_int[:, i] = np.interp(t_int, t, x[:, i])
        # quaternion interpolation
        t_int = torch.Tensor(t_int - t[0])
        t = torch.Tensor(t - t[0])
        qs = SO3.qnorm(torch.Tensor(x[:, 4:8]))
        x_int[:, 4:8] = SO3.qinterp(qs, t, t_int).numpy()
        return x_int

    def read_data(self, data_dir):
        print("Reading data ...")
        for sequence in self.sequences:
            print("Sequence name: " + sequence)
            imu_path = os.path.join(data_dir, sequence, "csv","imu_data.csv")
            gt_path = os.path.join(data_dir, sequence, "csv", "ground_truth.csv")
            imu = np.genfromtxt(imu_path, delimiter=",", skip_header=1)
            gt = np.genfromtxt(gt_path, delimiter=",", skip_header=1)
            t_start = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])
            index_start_imu = np.searchsorted(imu[:, 0], t_start)
            index_start_gt = np.searchsorted(gt[:, 0], t_start)
            index_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            index_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')
            imu = imu[index_start_imu: index_end_imu]
            gt = gt[index_start_gt: index_end_gt]

            ts = imu[:, 0]/1e9
            imu = torch.Tensor(imu[:, 1:]).double()                         # Cleaned imus

            gt = self.interpolate(gt, gt[:, 0]/1e9, ts)

            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0]
            p_gt = torch.Tensor(p_gt).double()                              # Cleaned p_gt

            q_gt = torch.Tensor(gt[:, 4:8]).double()
            q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)                    # Cleaned q_qt

            Rot_gt = SO3.from_quaternion(q_gt, ordering='wxyz').cpu()
            dRot_ij = bmtm(Rot_gt[:-self.min_train_freq], Rot_gt[self.min_train_freq:])
            dRot_ij = SO3.dnormalize(dRot_ij.cuda())
            dxi_ij = SO3.log(dRot_ij)

            dp_ij = - p_gt[:-self.min_train_freq] + p_gt[self.min_train_freq:]
            
            self.sequence_data[sequence] =  {
                't': ts,
                'xs': dxi_ij.float().cuda(),
                'dp': dp_ij.float().cuda(),
                'imu': imu.float().cuda(),
                'p': p_gt.float().cuda(),
                'q': q_gt.float().cuda(),
            }

