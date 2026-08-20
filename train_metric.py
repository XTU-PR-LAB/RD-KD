
from __future__ import absolute_import, division, print_function

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from trainer_metric import Trainermetric
from options_lite import LiteMonoOptions
from logger import Logger

options = LiteMonoOptions()
opts = options.parse()
print(">>> model from args:", opts.model)


def setup(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, opts):
    logger = None
    if rank == 0:
        log_dir = os.path.join(opts.log_dir, opts.model_name, "logs")
        logger = Logger(log_dir)
        logger.enable()
    
    if world_size > 1:
        setup(rank, world_size)
    
    opts.rank = rank
    opts.world_size = world_size
    
    if rank == 0:
        print(f"启动训练，使用{world_size}个进程")
    

    try:
        trainer = Trainermetric(opts)
        trainer.train()
    finally:
        if rank == 0 and logger is not None:
            logger.disable()

        if world_size > 1 and dist.is_initialized():
            cleanup()


if __name__ == "__main__":
    if torch.cuda.device_count() > 1:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            main(rank, world_size, opts)
        else:
            world_size = torch.cuda.device_count()
            mp.spawn(main, args=(world_size, opts), nprocs=world_size, join=True)
    else:
        main(0, 1, opts)
