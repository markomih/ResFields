import os
import os.path as osp
import time
import logging
import argparse
import wandb
from datetime import datetime
import sys
sys.path.append(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'resfields'))

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
parser.add_argument('--gpu', default='-1', help='GPU(s) to be used. Set -1 to use all gpus.')
# parser.add_argument('--resume', default=None, help='path to the weights to be resumed')
# parser.add_argument('--resume_weights_only', action='store_true',
#     help='specify this argument to restore only the weights (w/o training states), e.g. --resume path/to/resume --resume_weights_only'
# )

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train', action='store_true')
group.add_argument('--validate', action='store_true')
group.add_argument('--test', action='store_true')
group.add_argument('--predict', action='store_true')
group.add_argument('--wandb_logger', action='store_true', default=False)

parser.add_argument('--exp_dir', default='../exp')
parser.add_argument('--verbose', action='store_true', help='if true, set logging level to DEBUG')
parser.add_argument("--model_ckpt", type=str, default=None, help="path to model checkpoint")
args, extras = parser.parse_known_args()

import datasets
import systems
import pytorch_lightning as pl
from utils.callbacks import ConfigSnapshotCallback, CustomProgressBar
from utils.misc import load_config    

def main():
    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    # config.trial_name = config.get('trial_name') or (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    config.exp_dir = config.get('exp_dir') or osp.join(args.exp_dir, config.name)
    config.save_dir = config.get('save_dir') or osp.join(config.exp_dir, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or osp.join(config.exp_dir, 'ckpt')
    config.config_dir = config.get('config_dir') or osp.join(config.exp_dir, 'config')

    logger = logging.getLogger('pytorch_lightning')
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if 'seed' not in config:
        config.seed = int(time.time() * 1000) % 1000
    pl.seed_everything(config.seed)

    dm = datasets.make(config.dataset.name, config.dataset)
    config.model.metadata = dm.get_metadata(config.dataset)
    system = systems.make(config.system.name, config)

    callbacks = []
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=config.ckpt_dir,  **config.checkpoint)
    callbacks.append(checkpoint_callback)
    if args.train:
        callbacks += [
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            ConfigSnapshotCallback(
                config, config.config_dir, use_version=False
            ),
            CustomProgressBar(refresh_rate=50),
        ]
    last_ckpt = os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.CHECKPOINT_NAME_LAST}.ckpt")
    if not os.path.exists(last_ckpt):
        last_ckpt = None
    if args.model_ckpt is not None:  # overwrite last ckpt if specified model path
        last_ckpt = args.model_ckpt
    resume_from_checkpoint = config.get('resume_from_checkpoint', last_ckpt)

    loggers = []
    if args.train:
        if args.wandb_logger:
            _logger = pl.loggers.WandbLogger(
                name=config.name,
                project='resfields',
                entity='markomih',
                save_dir=config.exp_dir,
                config=config,
                # offline=cfg.wandb.offline,
                settings=wandb.Settings(start_method='fork')
                )
        else:
            _logger = pl.loggers.TensorBoardLogger(osp.join(config.exp_dir, 'runs'), name=config.name)
        loggers.append(_logger)
    
    trainer = pl.Trainer(
        devices=n_gpus,
        accelerator='gpu',
        # accelerator='cpu',
        callbacks=callbacks,
        resume_from_checkpoint=resume_from_checkpoint,
        logger=loggers,
        # strategy='ddp',  # TIP: disable ddp for easier debugging with pdb
        strategy='ddp_find_unused_parameters_false',  # TIP: disable ddp for easier debugging with pdb
        # strategy='find_unused_parameters=True',  # TIP: disable ddp for easier debugging with pdb
        **config.trainer
    )

    if args.train:
        trainer.fit(system, datamodule=dm, ckpt_path=resume_from_checkpoint)
        trainer.test(system, datamodule=dm)
    elif args.validate:
        trainer.validate(system, datamodule=dm, ckpt_path=resume_from_checkpoint)
    elif args.test:
        trainer.test(system, datamodule=dm, ckpt_path=resume_from_checkpoint)
    elif args.predict:
        trainer.predict(system, datamodule=dm, ckpt_path=resume_from_checkpoint)

    # for _gpu in gpu_list:
    #     print(f'MAX MEMORY ALLOCATED (GPU#{_gpu})\t', int(torch.cuda.max_memory_allocated(int(_gpu))/(2**20)), 'MB')


if __name__ == '__main__':
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    ARGS_GPU = os.environ['CUDA_VISIBLE_DEVICES'] if args.gpu == '-1' else args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS_GPU
    gpu_list = ARGS_GPU.split(',')
    n_gpus = len(gpu_list)
    main()
