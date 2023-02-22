"""
Train a image encoder for condition information of diffusion model
"""
import argparse
import os
import warnings
import builtins
import torch
import sys

sys.path.append(".")
sys.path.append("..")

import torch.multiprocessing as mp
import torch.distributed as dist
from diffusion.script_util import (
    add_dict_to_argparser,
)
from coach import Coach


def main():
    args = create_argparser().parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU.This will completely disable dataset parallel.')

    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngous_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size need to be adjusted accordingly
        args.world_size = ngous_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngous_per_node, args=(ngous_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngous_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    torch.cuda.set_device(gpu)
    args.gpu = gpu

    # Suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print(f'Use GPU: {args.gpu} for training')

    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ['RANK'])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method="env://127.0.0.1:23456",
                                world_size=args.world_size, rank=args.rank)

    coach = Coach(args)
    coach.train()


def distributed_default():
    return dict(
        world_size=1,
        rank=1,
        dist_url="env://",
        dist_backend="nccl",
        gpu=None,
        multiprocessing_distributed=True,
    )


def create_argparser():
    defaults = dict(
        data_dir="/training_data/CelebA/celeba-hq-30000/celeba-256-tmp",
        exp_dir="/home/gongshuai/con-diffusion/20221108-attribute20-resnet18/",
        learning_rate=1e-5,
        batch_size=64,
        max_steps=300,
        save_interval=50,
        attribute_count=20,
    )
    defaults.update(distributed_default())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '23456'
    main()
