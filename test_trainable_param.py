'''
This is temp file that just loads the model to debug checkpoint loading
'''
import argparse
import datetime
import os
import os.path as osp
import time
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import build_loader, build_text_transform, imagenet_classes
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import collect_env, get_git_hash
from mmseg.apis import multi_gpu_test
from models import build_model
from omegaconf import OmegaConf, read_write
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
from timm.utils import AverageMeter, accuracy
from utils import (auto_resume_helper, build_dataset_class_tokens, build_optimizer, build_scheduler, data2cuda,
                   get_config, get_grad_norm, get_logger, load_checkpoint, parse_losses, reduce_tensor, save_checkpoint)
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None




def parse_args():
    parser = argparse.ArgumentParser('GroupViT training and evaluation script')
    parser.add_argument('--cfg', default='configs/gs3_group_vit_gcc_yfcc_30e.yml', type=str, required=True, help='path to config file')
    args = parser.parse_args()

    return args

def load_config(cfg_file, merge_base=True):
    from omegaconf import OmegaConf 
    cfg = OmegaConf.load(cfg_file)
    if '_base_' in cfg and merge_base:
        if isinstance(cfg._base_, str):
            base_cfg = OmegaConf.load(osp.join(osp.dirname(cfg_file), cfg._base_))
        else:
            base_cfg = OmegaConf.merge(OmegaConf.load(f) for f in cfg._base_)
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    print("cfg", cfg)

    if cfg.train.amp_opt_level != 'O0':
        assert amp is not None, 'amp not installed!'
    
    logger = get_logger()
    logger.info(f'Creating model:{cfg.model.type}/{cfg.model_name}')
    model = build_model(cfg.model)
    optimizer = build_optimizer(cfg.train, model)


if __name__ == '__main__':
    main()