# -*- coding: utf-8 -*-
# https://github.com/MorvanZhou/pytorch-A3C

import gym
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


class EvaluateNet(nn.Module):
    def __init__(self, o_dim, s_dim, a_dim):
        super(EvaluateNet, self).__init__()
        self.o_dim = o_dim
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.l1 = nn.Linear(o_dim, s_dim)
        self.l2 = nn.Linear(2*s_dim, 1)
        layer_init([self.l1, self.l2])

    def forward(self, o,s):
        s_bar = torch.tanh(self.l1(o))  
        lr = self.l2(torch.concat(s_bar, s))
        lr = torch.tanh(lr)
        return lr

class PolicyNet(nn.Module):
    def __init__(self, o_dim, s_dim, a_dim):
        super(PolicyNet, self).__init__()
        self.o_dim = o_dim
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(o_dim, s_dim)
        self.pi2 = nn.Linear(s_dim, a_dim)
        self.v1 = nn.Linear(o_dim, s_dim)
        self.v2 = nn.Linear(s_dim, 1)
        layer_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss