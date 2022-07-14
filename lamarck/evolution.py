
import argparse
import gym
import math
import numpy as np
import os
import random
import string
import torch
import torch.multiprocessing as multiprocessing

from model import PolicyNet, EvaluateNet

os.environ["OMP_NUM_THREADS"] = "1"

def evolution_net_mutate(item):
    # TODO:seed
    if not item['perturb']:
        return
    np.random.seed(1)
    for v in item['evolution_net'].state_dict().values():
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(0.05*eps).float()
    print("The evolution_net of {} is mutated.".format(item['name']))

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def generation_forward(item):
    name = item['name']
    env_name = item['env_name']
    policy_net = item['policy_net']
    evolution_net = item['evolution_net']
    steps = item['steps']
    env = gym.make(env_name).unwrapped
    opt = torch.optim.Adam(policy_net.parameters(), lr=1e-4, betas=(0.9, 0.999))
    gamma = 0.9
    r_sum = 0.
    step = 0
    while step < steps:
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        while True:
            # env.render()
            a = policy_net.choose_action(v_wrap(s[None, :]))
            s_, r, done, _ = env.step(a)
            if done: r = -1
            r_sum += r
            buffer_a.append(a)
            buffer_s.append(s)
            buffer_r.append(r)

            if step % 10 == 0 or done:
                if done:
                    v_s_ = 0.
                else:
                    v_s_ = policy_net.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]
                buffer_v_target = []
                for r in buffer_r[::-1]:
                    v_s_ = r + gamma * v_s_
                    buffer_v_target.append(v_s_)
                buffer_v_target.reverse()

                loss = policy_net.loss_func(
                    v_wrap(np.vstack(buffer_s)),
                    v_wrap(np.array(buffer_a), dtype=np.int64) if buffer_a[0].dtype == np.int64 else v_wrap(np.vstack(buffer_a)),
                    v_wrap(np.array(buffer_v_target)[:, None]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                buffer_s, buffer_a, buffer_r = [], [], []
                if done:
                    break
            s = s_
            step += 1
    return [name, r_sum]

class EvolutionManager(object):
    
    def __init__(self, args):
        self.args = args
        self.pool = multiprocessing.Pool(args.worker)
        env = gym.make(args.env_name)
        o_dim = env.observation_space.shape[0]
        s_dim = 128
        a_dim = env.action_space.n

        self.population_status = []
        for _ in range(args.population):
            individual_status = {}
            name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
            individual_status['name']= name
            env_name = self.args.env_name
            individual_status['env_name']= env_name
            policy_net = PolicyNet(o_dim, s_dim, a_dim)
            policy_net.share_memory()
            individual_status['policy_net']= policy_net
            evaluate_net = EvaluateNet(o_dim, s_dim, a_dim)
            evaluate_net.share_memory()
            individual_status['evolution_net']= evaluate_net
            steps = self.args.step_per_generation
            individual_status['steps']= steps
            self.population_status.append(individual_status)

    def __call__(self, generation_forward_fn, evolution_net_mutate, chunksize=1):
        for i in range(self.args.generation):
            print("===============generation {}===================".format(i))
            step_results = self.pool.map(generation_forward_fn, self.population_status, chunksize=chunksize)
            print(step_results)
            # get the middle postive value of rewards
            rewards = names = [result[1] for result in step_results]
            rewards_dict = {result[0]:result[1] for result in step_results}
            quantile = 50
            quantile_value = np.percentile(np.array(rewards), quantile)
            for individual in self.population_status:
                name = individual['name']
                individual['perturb'] = True if rewards_dict[name] > quantile_value else False
            mutate_results = self.pool.map(evolution_net_mutate, self.population_status,chunksize=chunksize)

def main(args):
    em = EvolutionManager(args)
    em(generation_forward, evolution_net_mutate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_name', type=str, default="CartPole-v0")
    parser.add_argument('-s', '--step_per_generation', type=int, default=100)
    parser.add_argument('-w', '--worker', type=int, default=4)
    parser.add_argument('-g', '--generation', type=int, default=10)
    parser.add_argument('-p', '--population', type=int, help='population size', default=4)
    args = parser.parse_args()
    main(args)
    