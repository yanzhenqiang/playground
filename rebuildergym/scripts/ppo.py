# -*- coding: future_fstrings -*-
# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py
# pip install future-fstrings 
# xvfb-run -a -s "-screen 0 1400x900x24" python3 xx.py 


from abc import ABC, abstractmethod
import argparse
from collections import OrderedDict
from copy import deepcopy
from distutils.util import strtobool
import inspect
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
import random
import os
import multiprocessing

import cloudpickle
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
# For old version,  use tensorboardX
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

# Define type aliases here to avoid circular import
VecEnvIndices = Union[None, int, Iterable[int]]
# VecEnvObs is what is returned by the reset() method
VecEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]
# VecEnvStepReturn contains the observation, reward, done, info for each env
VecEnvStepReturn = Tuple[VecEnvObs, np.ndarray, np.ndarray, List[Dict]]

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs: int, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self) -> VecEnvObs:
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: observation
        """
        raise NotImplementedError()

    @abstractmethod
    def step_async(self, actions: np.ndarray) -> None:
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        raise NotImplementedError()

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """
        Clean up the environment's resources.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """
        Return attribute from vectorized environment.

        :param attr_name: The name of the attribute whose value to return
        :param indices: Indices of envs to get attribute from
        :return: List of values of 'attr_name' in all environments
        """
        raise NotImplementedError()

    @abstractmethod
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """
        Set attribute inside vectorized environments.

        :param attr_name: The name of attribute to assign new value
        :param value: Value to assign to `attr_name`
        :param indices: Indices of envs to assign value
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """
        Call instance methods of vectorized environments.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: List of items returned by the environment's method call
        """
        raise NotImplementedError()

    @abstractmethod
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """
        Check if environments are wrapped with a given wrapper.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: True if the env is wrapped, False otherwise, for each env queried.
        """
        raise NotImplementedError()

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def get_images(self) -> Sequence[np.ndarray]:
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering

        :param mode: the rendering type
        """

        def tile_images(img_nhwc: Sequence[np.ndarray]) -> np.ndarray:
            """
            Tile N images into one big PxQ image
            (P,Q) are chosen to be as close as possible, and if N
            is square, then P=Q.

            :param img_nhwc: list or array of images, ndim=4 once turned into array. img nhwc
                n = batch index, h = height, w = width, c = channel
            :return: img_HWc, ndim=3
            """
            img_nhwc = np.asarray(img_nhwc)
            n_images, height, width, n_channels = img_nhwc.shape
            H = int(np.ceil(np.sqrt(n_images)))
            W = int(np.ceil(float(n_images) / H))
            img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, H * W)])
            out_image = img_nhwc.reshape((H, W, height, width, n_channels))
            out_image = out_image.transpose(0, 2, 1, 3, 4)
            out_image = out_image.reshape((H * height, W * width, n_channels))
            return out_image

        try:
            imgs = self.get_images()
        except NotImplementedError:
            return

        bigimg = tile_images(imgs)
        if mode == "human":
            import cv2  # pytype:disable=import-error
            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")

    @abstractmethod
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.

        :param seed: The random seed. May be None for completely random seeding.
        :return: Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        pass

    @property
    def unwrapped(self) -> "VecEnv":
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def getattr_depth_check(self, name: str, already_found: bool) -> Optional[str]:
        """Check if an attribute reference is being hidden in a recursive call to __getattr__

        :param name: name of attribute to check for
        :param already_found: whether this attribute has already been found in a wrapper
        :return: name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return f"{type(self).__module__}.{type(self).__name__}"
        else:
            return None

    def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices

class VecEnvWrapper(VecEnv):
    """
    Vectorized environment base class
    """

    def __init__(
        self,
        venv: VecEnv,
        observation_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
    ):
        self.venv = venv
        VecEnv.__init__(
            self,
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space,
        )
        self.class_attributes = dict(inspect.getmembers(self.__class__))

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self) -> VecEnvObs:
        pass

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        pass

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return self.venv.seed(seed)

    def close(self) -> None:
        return self.venv.close()

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.venv.render(mode=mode)

    def get_images(self) -> Sequence[np.ndarray]:
        return self.venv.get_images()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return self.venv.env_is_wrapped(wrapper_class, indices=indices)

    def __getattr__(self, name: str) -> Any:
        """Find attribute from wrapped venv(s) if this wrapper does not have it.
        Useful for accessing attributes from venvs which are wrapped with multiple wrappers
        which have unique attributes of interest.
        """
        blocked_class = self.getattr_depth_check(name, already_found=False)
        if blocked_class is not None:
            own_class = f"{type(self).__module__}.{type(self).__name__}"
            error_str = (
                f"Error: Recursive attribute lookup for {name} from {own_class} is "
                "ambiguous and hides attribute from {blocked_class}"
            )
            raise AttributeError(error_str)

        return self.getattr_recursive(name)

    def _get_all_attributes(self) -> Dict[str, Any]:
        """Get all (inherited) instance and class attributes

        :return: all_attributes
        """
        all_attributes = self.__dict__.copy()
        all_attributes.update(self.class_attributes)
        return all_attributes

    def getattr_recursive(self, name: str) -> Any:
        """Recursively check wrappers to find attribute.

        :param name: name of attribute to look for
        :return: attribute
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes:  # attribute is present in this wrapper
            attr = getattr(self, name)
        elif hasattr(self.venv, "getattr_recursive"):
            # Attribute not present, child is wrapper. Call getattr_recursive rather than getattr
            # to avoid a duplicate call to getattr_depth_check.
            attr = self.venv.getattr_recursive(name)
        else:  # attribute not present, child is an unwrapped VecEnv
            attr = getattr(self.venv, name)

        return attr

    def getattr_depth_check(self, name: str, already_found: bool) -> str:
        """See base class.

        :return: name of module whose attribute is being shadowed, if any.
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes and already_found:
            # this venv's attribute is being hidden because of a higher venv.
            shadowed_wrapper_class = f"{type(self).__module__}.{type(self).__name__}"
        elif name in all_attributes and not already_found:
            # we have found the first reference to the attribute. Now check for duplicates.
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, True)
        else:
            # this wrapper does not have the attribute. Keep searching.
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, already_found)

        return shadowed_wrapper_class

class CloudpickleWrapper(object):
    def __init__(self, var):
        """
        Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

        :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = cloudpickle.loads(obs)

def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'seed':
                remote.send(env.seed(data))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(data))
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError("`{}` is not implemented in the worker".format(cmd))
        except EOFError:
            break

class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: ([callable]) A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        # In some cases (like on GitHub workflow machine when running tests),
        # "forkserver" method results in an "connection error" (probably due to mpi)
        # We allow to bypass the default start method if an environment variable
        # is specified by the user
        if start_method is None:
            start_method = os.environ.get("DEFAULT_START_METHOD")

        # No DEFAULT_START_METHOD was specified, start_method may still be None
        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(('seed', seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', 'rgb_array'))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]


    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,  envs.action_space.n), std=0.01),
        )

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument('--gym-id', type=str, default="CartPole-v0")
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--total-timesteps', type=int, default=1000000)

    parser.add_argument('--n-minibatch', type=int, default=4)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--num-steps', type=int, default=128,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                         help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    args = parser.parse_args()

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.n_minibatch)

    # Setup the environment.
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # TODO:AttributeError: module 'tensorflow' has no attribute 'io'
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
            '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # Seeding
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def make_env(gym_id, seed, idx):
        def thunk():
            env = gym.make(gym_id)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    envs = VecPyTorch(
        SubprocVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)], "fork"),
        device
    )
    assert isinstance(envs.action_space, Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.anneal_lr:
        # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
        lr = lambda f: f * args.learning_rate

    # Storage for epoch data
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    for update in range(1, num_updates+1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = lr(frac)
            optimizer.param_groups[0]['lr'] = lrnow

        # Prepare the execution of the game.
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Put action logic here
            with torch.no_grad():
                values[step] = agent.get_value(obs[step]).flatten()
                action, logproba, _ = agent.get_action(obs[step])

            actions[step] = action
            logprobs[step] = logproba

            # Execute the game and log data.
            next_obs, rs, ds, infos = envs.step(action)
            rewards[step] = rs.view(-1)
            
            #  next_done = torch.Tensor(ds).to(device)
            #  TypeError: can't convert np.ndarray of type numpy.bool_.
            next_done = torch.Tensor(np.array(ds, dtype=np.uint8)).to(device)

            for info in infos:
                if 'episode' in info.keys():
                    print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                    writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                    break

        # bootstrap reward if not done. reached the batch limit
        with torch.no_grad():
            last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        nextvalues = values[t+1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        next_return = returns[t+1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,)+envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,)+envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizaing the policy and value network
        target_agent = Agent(envs).to(device)
        inds = np.arange(args.batch_size,)
        for i_epoch_pi in range(args.update_epochs):
            np.random.shuffle(inds)
            target_agent.load_state_dict(agent.state_dict())
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_ind = inds[start:end]
                mb_advantages = b_advantages[minibatch_ind]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                _, newlogproba, entropy = agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])
                ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

                # Stats
                approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                    v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.kle_stop:
                if approx_kl > args.target_kl:
                    break
            if args.kle_rollback:
                if (b_logprobs[minibatch_ind] - agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])[1]).mean() > args.target_kl:
                    agent.load_state_dict(target_agent.state_dict())
                    break

        # Record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        if args.kle_stop or args.kle_rollback:
            writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)

    envs.close()
    writer.close()
