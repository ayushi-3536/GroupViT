# -------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE
#
# Written by Ze Liu, Zhenda Xie
# Modified by Jiarui Xu
# -------------------------------------------------------------------------

from torch import optim as optim
from utils import get_logger

def check_items_in_string(items, key):
    for item in items:
        if item in key:
            return True
    return False

def set_gradient(model, cfg):
    #logger = get_logger()
    text_encoder_keys = ['text_encoder', 'text_projector']
    #Cross Attention module is added to take the cross attention between the group tokens
    #and text token before projecting group tokens into the multi-modal space 
    grouping_layer_key = ['.downsample', 'cross_attention.']
    for name, param in model.named_parameters():
        #logger.info(f'key: {name}')
        if not param.requires_grad:
            continue  # frozen weights
        if cfg.only_grouping:
            if not check_items_in_string(grouping_layer_key, name):
                param.requires_grad=False
        elif cfg.freeze_text_encoder:
            if check_items_in_string(text_encoder_keys, name):
                #logger.info(f'setting {name} as untrainable')
                param.requires_grad=False
    return model

def build_optimizer(config, model):
    """Build optimizer, set weight decay of normalization to 0 by default."""
    
    #logger = get_logger('optimizer')
    if config.finetune.only_grouping or config.finetune.freeze_text_encoder:
        #logger.info('Setting untrainable parameters')
        model = set_gradient(model, config.finetune)

    parameters = set_weight_decay(model, {}, {})
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #logger.info(f'Number of trainable parameters: {trainable_params}/{sum(p.numel() for p in model.parameters())}')

    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         logger.debug(f'name::{name}')

    opt_name = config.optimizer.name
    optimizer = None
    if opt_name == 'adamw':
        optimizer = optim.AdamW(
            parameters,
            eps=config.optimizer.eps,
            betas=config.optimizer.betas,
            lr=config.base_lr,
            weight_decay=config.weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer: {opt_name}')

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
