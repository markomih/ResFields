import argparse
import os
import time
import logging
from datetime import datetime
import torch
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
parser.add_argument('--gpu', default='-1', help='GPU(s) to be used. Set -1 to use all gpus.')
parser.add_argument('--resume', default=None, help='path to the weights to be resumed')
parser.add_argument('--resume_weights_only', action='store_true',
    help='specify this argument to restore only the weights (w/o training states), e.g. --resume path/to/resume --resume_weights_only'
)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train', action='store_true')
group.add_argument('--validate', action='store_true')
group.add_argument('--test', action='store_true')
group.add_argument('--predict', action='store_true')

parser.add_argument('--exp_dir', default='./exp')
parser.add_argument('--runs_dir', default='./runs')
parser.add_argument('--verbose', action='store_true', help='if true, set logging level to DEBUG')

args, extras = parser.parse_known_args()

# set CUDA_VISIBLE_DEVICES then import pytorch-lightning
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
ARGS_GPU = os.environ['CUDA_VISIBLE_DEVICES'] if args.gpu == '-1' else args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = ARGS_GPU
gpu_list = ARGS_GPU.split(',')
n_gpus = len(gpu_list)

import datasets
import systems
import pytorch_lightning as pl
from utils.callbacks import ConfigSnapshotCallback, CustomProgressBar
from utils.callbacks import NerfplayerCallback
from utils.misc import load_config    

def main():
    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    config.trial_name = config.get('trial_name') or (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    config.exp_dir = config.get('exp_dir') or os.path.join(args.exp_dir, config.name)
    config.save_dir = config.get('save_dir') or os.path.join(config.exp_dir, config.trial_name, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.config_dir = config.get('config_dir') or os.path.join(config.exp_dir, config.trial_name, 'config')

    logger = logging.getLogger('pytorch_lightning')
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if 'seed' not in config:
        config.seed = int(time.time() * 1000) % 1000
    pl.seed_everything(config.seed)

    dm = datasets.make(config.dataset.name, config.dataset)
    config.model.metadata = dm.get_metadata(config.dataset)
    system = systems.make(config.system.name, config, load_from_checkpoint=None if not args.resume_weights_only else args.resume)

    callbacks = []
    if args.train:
        callbacks += [
            pl.callbacks.ModelCheckpoint(
                dirpath=config.ckpt_dir,
                **config.checkpoint
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            ConfigSnapshotCallback(
                config, config.config_dir, use_version=False
            ),
            CustomProgressBar(refresh_rate=50),
        ]

    if('nerfplayer-ngp' in config.name):
        callbacks += [ 
            NerfplayerCallback()
        ]

    loggers = []
    if args.train:
        loggers += [
            pl.loggers.TensorBoardLogger(args.runs_dir, name=config.name, version=config.trial_name),
        ]
    
    trainer = pl.Trainer(
        devices=n_gpus,
        accelerator='gpu',
        # accelerator='cpu',
        callbacks=callbacks,
        logger=loggers,
        # strategy='ddp',  # TIP: disable ddp for easier debugging with pdb
        strategy='ddp_find_unused_parameters_false',  # TIP: disable ddp for easier debugging with pdb
        # strategy='find_unused_parameters=True',  # TIP: disable ddp for easier debugging with pdb
        **config.trainer
    )

    if args.train:
        if args.resume and not args.resume_weights_only:
            trainer.fit(system, datamodule=dm, ckpt_path=args.resume)
        else:
            trainer.fit(system, datamodule=dm)
        trainer.test(system, datamodule=dm)
    elif args.validate:
        trainer.validate(system, datamodule=dm, ckpt_path=args.resume)
    elif args.test:
        trainer.test(system, datamodule=dm, ckpt_path=args.resume)
    elif args.predict:
        trainer.predict(system, datamodule=dm, ckpt_path=args.resume)

    # for _gpu in gpu_list:
    #     print(f'MAX MEMORY ALLOCATED (GPU#{_gpu})\t', int(torch.cuda.max_memory_allocated(int(_gpu))/(2**20)), 'MB')


if __name__ == '__main__':
    main()
