import os
import random

import numpy as np
import torch

from models.model import AWN


def fix_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_exp_settings(logger, cfg):
    """
    log the current experiment settings.
    """
    logger.info('=' * 20)
    log_dict = cfg.__dict__.copy()
    for k, v in log_dict.items():
        logger.info(f'{k} : {v}')
    logger.info('=' * 20)


def create_AWN_model(cfg):
    """
    build AWN model
    """
    model = AWN(
        num_classes=cfg.num_classes,
        num_levels=cfg.num_level,
        in_channels=cfg.in_channels,
        kernel_size=cfg.kernel_size,
        latent_dim=cfg.latent_dim,
        regu_details=cfg.regu_details,
        regu_approx=cfg.regu_approx,
    ).to(cfg.device)

    return model
