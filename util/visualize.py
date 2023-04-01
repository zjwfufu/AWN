from functools import reduce

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os


def Visualize_LiftingScheme(model, signal, cfg, index):
    def hook_feature_map(module, *input_feature_map):
        if len(feature_map) < 3:
            feature_map.append(input_feature_map[0][0])
        feature_map.append(input_feature_map[1][0])  # approx
        feature_map.append(input_feature_map[1][1])  # details

    feature_map = []

    for i in range(0, cfg.num_level):
        # levels.level_{i}.wavelet.register_forward_hook
        # (hook_feature_map)
        suffix = f'levels.level_{i}.wavelet.register_forward_hook'
        split = suffix.split('.')
        reduce(getattr, split, model)(hook=hook_feature_map)

    model.to('cpu')
    model(signal)

    fig1 = plt.figure(figsize=(6, 10))
    gs = GridSpec(cfg.num_level + 1, 2, figure=fig1)

    for i in range(0, len(feature_map)):

        if i == 0:
            ax = fig1.add_subplot(gs[i, :2])
            ax.plot(feature_map[i].detach().numpy()[0, 0, :])
            ax.set_title(fr"$V_0$")
        else:
            ax = fig1.add_subplot(gs[(i + 1) // 2, (i + 1) % 2])
            ax.plot(feature_map[i].detach().numpy()[0, 0, :])
            if (i + 1) % 2 == 0:
                ax.set_title(fr'$L_{(i + 1) // 2}$')
            else:
                ax.set_title(fr'$H_{(i + 1) // 2}$')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    fig1.savefig(cfg.result_dir + '/' + f'visualize_one_channel_#{index}.svg', format='svg', dpi=150)

    fig2 = plt.figure(figsize=(6, 10))
    gs = GridSpec(cfg.num_level + 1, 2, figure=fig2)

    for i in range(0, len(feature_map)):

        if i == 0:
            ax = fig2.add_subplot(gs[i, :2])
            ax.imshow(feature_map[i].detach().numpy()[0], aspect='auto', cmap='RdBu')
            ax.set_title(fr"$V_0$")
        else:
            ax = fig2.add_subplot(gs[(i + 1) // 2, (i + 1) % 2])
            ax.imshow(feature_map[i].detach().numpy()[0], aspect='auto', cmap='RdBu')
            if (i + 1) % 2 == 0:
                ax.set_title(fr'$L_{(i + 1) // 2}$')
            else:
                ax.set_title(fr'$H_{(i + 1) // 2}$')
        ax.set_yticks([])

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    fig2.savefig(cfg.result_dir + '/' + f'visualize_feature_map_#{index}.svg', format='svg', dpi=150)

    plt.close()


def Draw_Confmat(Confmat_Set, snrs, cfg):
    for i, snr in enumerate(snrs):
        fig = plt.figure()
        df_cm = pd.DataFrame(Confmat_Set[i],
                             index=cfg.classes,
                             columns=cfg.classes)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        heatmap.yaxis.set_ticklabels(
            heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(
            heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        conf_mat_dir = os.path.join(cfg.result_dir, 'conf_mat')
        os.makedirs(conf_mat_dir, exist_ok=True)
        fig.savefig(conf_mat_dir + '/' + f'ConfMat_{snr}dB.svg', format='svg', dpi=150)
        plt.close()


def Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg):
    plt.plot(snrs, Accuracy_list)
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Overall Accuracy")
    plt.title(f"Overall Accuracy on {cfg.dataset} dataset")
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid()
    acc_dir = os.path.join(cfg.result_dir, 'acc')
    os.makedirs(acc_dir, exist_ok=True)
    plt.savefig(acc_dir + '/' + 'acc.svg', format='svg', dpi=150)
    plt.close()

    Accuracy_Mods = np.zeros((len(snrs), Confmat_Set.shape[-1]))

    for i, snr in enumerate(snrs):
        Accuracy_Mods[i, :] = np.diagonal(Confmat_Set[i]) / Confmat_Set[i].sum(1)

    for j in range(0, Confmat_Set.shape[-1]):
        plt.plot(snrs, Accuracy_Mods[:, j])

    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Overall Accuracy")
    plt.title(f"Overall Accuracy on {cfg.dataset} dataset")
    plt.grid()
    plt.legend(cfg.classes.keys())
    plt.savefig(acc_dir + '/' + 'acc_mods.svg', format='svg', dpi=150)
    plt.close()


def save_training_process(train_process, cfg):
    fig1 = plt.figure(1)
    plt.plot(train_process.epoch, train_process.lr_list)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("learning rate")
    plt.grid()
    fig1.savefig(cfg.result_dir + '/' + 'lr.svg', format='svg', dpi=150)

    fig2 = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss,
             "ro-", label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss,
             "bs-", label="Val loss")
    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc,
             "ro-", label="Train acc")
    plt.plot(train_process.epoch, train_process.val_acc,
             "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.grid()
    fig2.savefig(cfg.result_dir + '/' + 'loss_acc.svg', format='svg', dpi=150)
    plt.show()
