import numpy as np
import torch
from utils import bmtm, bmtv, bmmt, SO3


class GyroNet(torch.nn.Module):
    def __init__(self, channel, dropout, kernel, dilation, momentum):
        super().__init__()
        pad = (kernel[0]-1) + dilation[1]*(kernel[1]-1) + dilation[2]*(kernel[2]-1) +  dilation[3]*(kernel[3]-1)
        self.cnn = torch.nn.Sequential(
            torch.nn.ReplicationPad1d((pad, 0)),
            torch.nn.Conv1d(channel[0], channel[1], kernel[0], dilation=dilation[0]),
            torch.nn.BatchNorm1d(channel[1], momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(channel[1], channel[2], kernel[1], dilation=dilation[1]),
            torch.nn.BatchNorm1d(channel[2], momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(channel[2], channel[3], kernel[2], dilation=dilation[2]),
            torch.nn.BatchNorm1d(channel[3], momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(channel[3], channel[4], kernel[3], dilation=dilation[3]),
            torch.nn.BatchNorm1d(channel[4], momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(channel[4], channel[5], kernel[4], dilation=dilation[4]),
            torch.nn.ReplicationPad1d((0, 0)),
        )
        self.mean_u = torch.nn.Parameter(torch.zeros(channel[0]),
            requires_grad=False)
        self.std_u = torch.nn.Parameter(torch.ones(channel[0]),
            requires_grad=False)
        self.gyro_std = torch.nn.Parameter(torch.Tensor([1*np.pi/180, 2*np.pi/180, 5*np.pi/180]), requires_grad=False)
        self.gyro_Rot = torch.nn.Parameter(0.05*torch.randn(3, 3).cuda())

    def forward(self, imus):
        # imus_c: w_1, w_2, w_3, a_1, a_2, a_3
        # imus: b l c --> b c l
        imus_normal = ((imus-self.mean_u)/self.std_u).transpose(1, 2)

        # hat_xs: b c l = 1 3 16000
        hat_xs = self.cnn(imus_normal)
        Rots = (torch.eye(3).cuda() + self.gyro_Rot).expand(imus.shape[0], imus.shape[1], 3, 3)
        Rot_imus = torch.einsum('baij, baj -> bai', Rots, imus[:, :, :3])
        hat_xs = self.gyro_std*hat_xs.transpose(1, 2) + Rot_imus
        return hat_xs

    def set_normalized_factors(self, mean_u, std_u):
        self.mean_u = torch.nn.Parameter(torch.FloatTensor(mean_u).cuda(), requires_grad=False)
        self.std_u = torch.nn.Parameter(torch.FloatTensor(std_u).cuda(), requires_grad=False)


class GyroLoss(torch.nn.Module):
    """Loss for low-frequency orientation increment"""

    def __init__(self, w, min_N, max_N, dt, target, huber):
        super().__init__()
        self.min_N = min_N
        self.max_N = max_N
        self.min_train_freq = 2 ** self.min_N
        self.max_train_freq = 2 ** self.max_N
        self.dt = dt
        self.w = w
        self.sl = torch.nn.SmoothL1Loss()
        self.huber = huber
        self.weight = torch.ones(1, 1, self.min_train_freq).cuda()/self.min_train_freq
        self.N0 = 5 # remove first N0 increment in loss due not account padding

    def f_huber(self, rs):
        loss = self.w*self.sl(rs/self.huber,
            torch.zeros_like(rs))*(self.huber**2)
        return loss


    def forward(self, xs, dp, hat_xs):
        N = xs.shape[0]
        xs = SO3.exp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.exp(hat_xs[:, :3])
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])

        rs = SO3.log(bmtm(Omegas, xs)).reshape(N, -1, 3)[:, self.N0:]
        # b l c
        loss = self.f_huber(rs)
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            xs = xs[::2].bmm(xs[1::2])
            rs = SO3.log(bmtm(Omegas, xs)).reshape(N, -1, 3)[:, self.N0:]

            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss
