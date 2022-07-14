CONFIGCODE ADD X3PD3C,7PKM76,D48WJD,GJ3HBM,9KFTM4,HWCT_70_0001


# -*- coding: utf-8 -*-
# https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease/WANN
# https://github.com/atgambardella/pytorch-es
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
import argparse
import os
import subprocess
import sys
import time

from mpi4py import MPI

from model import PopulationManager, Task


def mpi_fork(n):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    (from https://github.com/garymcintire/mpi_util/)
    """

    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            IN_MPI="1"
        )
        print(["mpirun", "-np", str(n), sys.executable] + sys.argv)
        subprocess.check_call(["mpirun", "-np", str(n), sys.executable]
                              + ['-u'] + sys.argv, env=env)
        return "parent"
    else:
        return "child"


def main(args):
    rank = MPI.COMM_WORLD.Get_rank()
    if (rank == 0):
        master(args)
    else:
        slave(args)


def master(args):
    pm = PopulationManager()
    for generation in range(args.generation]):        
        population_start_status = pm.start_generation()
        time.sleep(1000)
        population_end_status = pm.stop_generation()
        reward = pm.eval()
    stop_workers()


def slave(args):
    task = Task()
    while True:
        cmd = MPI.COMM_WORLD.recv(source=0,  tag=1)
    if cmd == -1:
        print('Worker # ', rank, ' shutting down.')
        break
    if cmd == 1:
        pass


def stop_workers():
  num_of_slave = MPI.COMM_WORLD.Get_size() - 1
  print('Stopping workers')
  for i_work in range(num_of_slave):
    MPI.COMM_WORLD.send(-1, dest=(i_work)+1, tag=1)

if __name__ == "__main__":
    # Parse argument.
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--windows', type=int, help="TODO", default=1000)
    parser.add_argument('-n', '--num_worker', type=int, help='number of cores to use', default=4)
    parser.add_argument('-g', '--generation', type=int, help='generation of evolution', default=4)
    args = parser.parse_args()

    # Use MPI if parallel
    if "parent" == mpi_fork(args.num_worker+1): os._exit(0)
    main(args)