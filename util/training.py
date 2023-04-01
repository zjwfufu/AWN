import os.path
import time

import pandas as pd
import torch
from torch import optim, nn
from tqdm import tqdm

from util.early_stop import EarlyStopping
from util.evaluation import Run_Eval
from util.logger import AverageMeter


class Trainer:
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 cfg,
                 logger):
        super(Trainer, self).__init__()

        self.epochs_stats = None
        self.val_acc_list = None
        self.val_loss_list = None
        self.train_acc_list = None
        self.train_loss_list = None
        self.val_acc = None
        self.val_loss = None
        self.train_acc = None
        self.best_monitor = None
        self.lr_list = None
        self.train_loss = None
        self.t_s = None
        self.early_stopping = None
        self.criterion = None
        self.optimizer = None

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger

        self.iter = 0

    def loop(self):
        self.before_train()
        for self.iter in range(0, self.cfg.epochs):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()
            if self.early_stopping.early_stop:
                self.logger.info('Early stopping')
                break

    @staticmethod
    def adjust_learning_rate(optimizer, gamma):
        """Sets the learning rate when we have to"""
        lr = optimizer.param_groups[0]['lr'] * gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.cfg.device)
        self.early_stopping = EarlyStopping(self.logger, patience=self.cfg.patience)

        self.lr_list = []
        self.best_monitor = 0.0
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def before_train_step(self):
        self.model.train()
        self.t_s = time.time()
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):
        with tqdm(total=len(self.train_loader),
                  desc=f'Epoch{self.iter}/{self.cfg.epochs}',
                  postfix=dict,
                  mininterval=0.3) as pbar:
            for step, (sig_batch, lab_batch) in enumerate(self.train_loader):
                sig_batch = sig_batch.to(self.cfg.device)
                lab_batch = lab_batch.to(self.cfg.device)

                logit, regu_sum = self.model(sig_batch)

                loss = self.criterion(logit, lab_batch)
                loss += sum(regu_sum)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pre_lab = torch.argmax(logit, 1)
                acc = torch.sum(pre_lab == lab_batch.data).double().item() / lab_batch.size(0)

                self.train_loss.update(loss.item())
                self.train_acc.update(acc)

                pbar.set_postfix(**{'train_loss': self.train_loss.avg,
                                    'train_acc': self.train_acc.avg})
                pbar.update(1)

    def after_train_step(self):
        self.lr_list.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} Train acc: {} lr: {:.5f}'.format(self.iter,
                                                                                          time.time() - self.t_s,
                                                                                          self.train_loss.avg,
                                                                                          self.train_acc.avg,
                                                                                          self.lr_list[-1]))
        self.train_loss_list.append(self.train_loss.avg)
        self.train_acc_list.append(self.train_acc.avg)

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_loss = AverageMeter()
        self.val_acc = AverageMeter()
        self.logger.info(f"Starting validation epoch {self.iter}:")

    def run_val_step(self):
        with tqdm(total=len(self.val_loader),
                  desc=f'Epoch{self.iter}/{self.cfg.epochs}',
                  postfix=dict,
                  mininterval=0.3,
                  colour='blue') as pbar:
            for step, (sig_batch, lab_batch) in enumerate(self.val_loader):
                with torch.no_grad():
                    sig_batch = sig_batch.to(self.cfg.device)
                    lab_batch = lab_batch.to(self.cfg.device)

                    logit, regu_sum = self.model(sig_batch)

                    loss = self.criterion(logit, lab_batch)
                    loss += sum(regu_sum)

                    pre_lab = torch.argmax(logit, 1)
                    acc = torch.sum(pre_lab == lab_batch.data).double().item() / lab_batch.size(0)

                    self.val_loss.update(loss.item())
                    self.val_acc.update(acc)

                    pbar.set_postfix(**{'val_loss': self.val_loss.avg,
                                        'val_acc': self.val_acc.avg})
                    pbar.update(1)

    def after_val_step(self):
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Val Loss: {} Val acc: {}'.format(self.iter,
                                                                           time.time() - self.t_s,
                                                                           self.val_loss.avg,
                                                                           self.val_acc.avg))
        if self.cfg.monitor == 'acc':
            if self.val_acc.avg >= self.best_monitor:
                self.best_monitor = self.val_acc.avg
                save_model_name = self.cfg.dataset + '_' + 'AWN' + '.pkl'
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_dir, save_model_name))
        else:
            raise NotImplementedError(f'Not Implement monitor: {self.cfg.monitor}')

        self.early_stopping(self.val_loss.avg, self.model)

        if self.early_stopping.counter !=0 and self.early_stopping.counter % self.cfg.milestone_step == 0:
            self.adjust_learning_rate(self.optimizer, self.cfg.gamma)

        self.val_loss_list.append(self.val_loss.avg)
        self.val_acc_list.append(self.val_acc.avg)

        self.epochs_stats = pd.DataFrame(
            data={"epoch": range(self.iter + 1),
                  "lr_list": self.lr_list,
                  "train_loss": self.train_loss_list,
                  "val_loss": self.val_loss_list,
                  "train_acc": self.train_acc_list,
                  "val_acc": self.val_acc_list}
        )


