import argparse
import os.path

import numpy as np
import torch

from data_loader.data_loader import Create_Data_Loader, Load_Dataset, Dataset_Split
from util.config import Config, merge_args2cfg
from util.evaluation import Run_Eval
from util.training import Trainer
from util.utils import fix_seed, log_exp_settings, create_AWN_model
from util.logger import create_logger
from util.visualize import Visualize_LiftingScheme, save_training_process

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')  # train ,eval or visualize
    parser.add_argument('--dataset', type=str, default='2016.10a')  # 2016.10a, 2016.10b, 2018.01a
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--Draw_Confmat', type=bool, default=True)
    parser.add_argument('--Draw_Acc_Curve', type=bool, default=True)
    args = parser.parse_args()

    fix_seed(args.seed)

    cfg = Config(args.dataset, train=(args.mode == 'train'))
    cfg = merge_args2cfg(cfg, vars(args))
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    log_exp_settings(logger, cfg)

    model = create_AWN_model(cfg)
    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    Signals, Labels, SNRs, snrs, mods = Load_Dataset(cfg.dataset, logger)
    train_set, test_set, val_set, test_idx = Dataset_Split(
        Signals,
        Labels,
        snrs,
        mods,
        logger)
    Signals_test, Labels_test = test_set

    if args.mode == 'train':
        train_loader, val_loader = Create_Data_Loader(train_set, val_set, cfg, logger)
        trainer = Trainer(model,
                          train_loader,
                          val_loader,
                          cfg,
                          logger)
        trainer.loop()
        
        save_training_process(trainer.epochs_stats, cfg)

        save_model_name = cfg.dataset + '_' + 'AWN' + '.pkl'
        model.load_state_dict(torch.load(os.path.join(cfg.model_dir, save_model_name)))
        Run_Eval(model,
                 Signals_test,
                 Labels_test,
                 SNRs,
                 test_idx,
                 cfg,
                 logger)

    elif args.mode == 'eval':
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, cfg.dataset + '_' + 'AWN' + '.pkl')))
        Run_Eval(model,
                 Signals_test,
                 Labels_test,
                 SNRs,
                 test_idx,
                 cfg,
                 logger)

    elif args.mode == 'visualize':
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, cfg.dataset + '_' + 'AWN' + '.pkl')))
        for i in range(0, 8):
            index = np.random.randint(0, Signals_test.shape[0])
            test_sample = Signals_test[index]
            test_sample = test_sample[np.newaxis, ...]
            Visualize_LiftingScheme(model, test_sample, cfg, index)
