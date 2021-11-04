#! /usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from mpl_toolkits import mplot3d

import sys
import numpy as np
import pandas as pd
import torch


def plot_losses(train_losses, valid_losses, test_loss=None, save_fname=None):
    """
    To be made more flexible.
    """
    
    train_losses_classification, train_losses_regression, _ = np.array(train_losses).T
    valid_losses_classification, valid_losses_regression, _ = np.array(valid_losses).T
    if test_loss is not None:
        test_loss_classification, test_loss_regression = test_loss

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)
    
    # Classification
    axes[0][0].plot(train_losses_classification, label='train', color='teal')
    axes[0][1].plot(valid_losses_classification, label='validation', color='teal')
    axes[0][0].set_title('Train', fontsize=20)
    axes[0][1].set_title('Validation', fontsize=20)
    for i, ax in enumerate(axes[0]):
        if i == 0:
            ax.set_ylabel('classification loss', fontsize=16)
        if test_loss is not None:
            ax.axhline(test_loss_classification, label='test', color='purple', linewidth=3)

    # Regression
    axes[1][0].plot(train_losses_regression, label='train', color='blue')
    axes[1][1].plot(valid_losses_regression, label='validation', color='blue')
    for i, ax in enumerate(axes[1]):
        if i == 0:
            ax.set_ylabel('regression loss', fontsize=16)
        ax.set_xlabel('batches', fontsize=16)
        if test_loss is not None:
            ax.axhline(test_loss_regression, label='test', color='orange', linewidth=3)

    for ax in axes.flatten():
        ax.tick_params(axis='both', labelsize=12)
        ax.legend()

    fig.subplots_adjust(hspace=0.2)
    fig.tight_layout()
    
    if save_fname is not None:
        fig.savefig(save_fname, transparent=False, bbox_inches='tight', dpi=200)
    plt.show()
    plt.close(fig)
    

def plot_histograms(X, Y_c, Y, M=1000000, save_fname=None):
    """
    X, Y_c, and Y all have to be one-dimensional numpy arrays
    X: flattened ground truth values
    Y_c: flattened predicted classes
    Y: flattend predicted values
    """
    
    overall_mse = ((Y - X) ** 2).mean()
    print(f'overall mean squared error = {overall_mse: .3f}')
    mse = np.sum((Y - X)**2 * (X > 0)) / np.sum(X > 0)
    print(f'non-zero mean squared error = {mse: .3f}')

    if len(X) > M:
        indices = np.random.choice(len(X), M, replace=False)
        X, Y_c, Y = X[indices], Y_c[indices], Y[indices]
                    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 4), sharey=False)
    facecolor, edgecolor = 'royalblue', 'aliceblue'
    
    axes[0].hist(Y_c, bins=np.linspace(0, 1, 51), label='predicted class', facecolor=facecolor, edgecolor=edgecolor, zorder=3, alpha=.8)
    axes[1].hist(np.log2(X + 1), bins=np.linspace(0, 10, 51), label='ground truth', facecolor=facecolor, edgecolor=edgecolor, zorder=3, alpha=.8)
    axes[2].hist(np.log2(Y + 1), bins=np.linspace(0, 10, 51), label='decompressed', facecolor=facecolor, edgecolor=edgecolor, zorder=3, alpha=.8)
    for ax in axes.flatten():
        ax.grid('True', which='both', zorder=0)
        ax.set_yscale('log')
        ax.set_ylim(1e2, 1e6)
        ax.tick_params(axis='both', labelsize=15)
        ax.legend(fontsize=18)
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=.1)
    if save_fname is not None:
        fig.savefig(save_fname, transparent=False, bbox_inches='tight', dpi=200)
    plt.show()
    plt.close(fig)


def plot_errors(x, y_c, save_fname=None):
    """
    X_c and Y_c both have to be one-dimensional numpy arrays
    X_c: flattened ground truth labels
    Y_c: flattened predicted labels
    """

    X_c = torch.log2(x + 1).cpu().detach().numpy().flatten() > 1
    Y_c = y_c.cpu().detach().numpy().flatten()
    
    pos = X_c.sum()
    neg = len(X_c) - pos

    T = np.linspace(0, 1, 101)
    FNR, FPR, ERR = [], [], []
    for t in T:
        tp, tn = ((Y_c > t) * X_c).sum(), ((Y_c < t) * (~X_c)).sum()
        fn, fp = pos - tp, neg - tn
        fnr, fpr = fn / pos, fp / neg
        error_rate = (fn + fp) / len(X_c)

        FNR.append(fnr)
        FPR.append(fpr)
        ERR.append(error_rate)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(T, FNR, label='false negative rate')
    ax.plot(T, FPR, label='false postive rate')
    ax.plot(T, ERR, label='error rate')
    ax.legend()
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xlabel('threshold', fontsize=16)
    ax.set_ylabel('error rate', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    plt.grid('True')
    plt.title('Errors', fontsize=18)
    
    fig.tight_layout()
    if save_fname is not None:
        fig.savefig(save_fname, transparent=False, bbox_inches='tight', dpi=200)
    plt.show()
    plt.close(fig)
    
    return pd.DataFrame(data={'FNR': FNR, 'FPR': FPR, 'ERR': ERR, 'threshold': T})
        
        
def plot_histogram_2d(X_log, Y_log, M=1000000, gamma=.2, save_fname=None):

    # Sampling for speedy plot
    if len(X_log) > M:
        indices = np.random.choice(len(X_log), M, replace=False)
        X, Y = X_log[indices], Y_log[indices]
    else:
        X, Y = X_log, Y_log
        
    data = np.array([X, Y]).T
    
    bins_x = np.linspace(6, 10, 81)
    y_bound = int(Y.max() * 10 + .5) / 10 + 0.05
    bins_y = np.arange(6, y_bound, .05)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    h, xedges, yedges, _ = ax.hist2d(
        data[:, 0], 
        data[:, 1], 
        bins=[bins_x, bins_y], 
        norm=mcolors.PowerNorm(gamma)
    )
    ax.set_aspect(1)
    # ax.set_xticks([6, 7, 8, 9, 10])
    # ax.set_yticks([6, 7, 8, 9, 10])
    x_min = max(X.min(), Y.min())
    # ax.set_xlim(x_min, 10)
    # ax.set_ylim(x_min, 10)
    ax.set_xlabel('ground truth', fontsize=16)
    ax.set_ylabel('prediction', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)

    if save_fname is not None:
        fig.savefig(save_fname, transparent=False, bbox_inches='tight', dpi=200)
    plt.show()
    plt.close(fig)
    
    return data, h, xedges, yedges
    

def visualize_2d(image_original, image_recovered, frame_axis=0, max_frames=5, figure_width=10, cmap='gray', save_fname=None, aspect=None):
    """
    The oringal data should be (azimuth, z, layer)
    """
    axis_labels = np.array(['azimuth', 'z', 'layer'])
    axis_labels = np.delete(axis_labels, frame_axis)

    shape = image_original.shape
    frame_indices = np.random.choice(shape[frame_axis], max_frames)
    
    image_original = image_original.cpu().detach().numpy()
    image_original = np.log2(image_original + 1)
    image_original = np.moveaxis(image_original, frame_axis, 0)
    
    image_recovered = image_recovered.cpu().detach().numpy()
    image_recovered = np.log2(image_recovered + 1)
    image_recovered = np.moveaxis(image_recovered, frame_axis, 0)

    a, b = image_original.shape[1:]
    if a > b:
        image_original = np.moveaxis(image_original, 1, 2)
        image_recovered = np.moveaxis(image_recovered, 1, 2)
        axis_labels = axis_labels[::-1]
    
    max_frames = min(image_original.shape[0], max_frames)
    
    h, w = image_original.shape[1:]
    if aspect is None:
        aspect = 1 - .031 * w / h
    H, W = h * max_frames, w * 2 * aspect
    figsize = (figure_width, figure_width * (H / W))
    fig, axes = plt.subplots(max_frames, 2, figsize=figsize, sharex=True, sharey=True)
    # print(type(image_recovered[frame_indices[0]][0, 0]))
    
    for i, index in enumerate(frame_indices): 
        if i == 0:
            axes[i][0].set_title('Original', fontsize=20)
            axes[i][1].set_title('Decompressed', fontsize=20)
        # axes[i][0].set_aspect(aspect)
        # axes[i][0].set_aspect(aspect)
        axes[i][0].imshow(image_original[index], cmap=cmap, interpolation='none')
        axes[i][1].imshow(image_recovered[index], cmap=cmap, interpolation='none')
    for ax in axes.flatten():
        ax.tick_params(axis='both', labelsize=12)
    
    fig.text(0.5, 0, axis_labels[1], va='top', ha='center', fontsize=16)
    fig.text(0, 0.5, axis_labels[0], va='center', ha='right', rotation='vertical', fontsize=16)
    fig.tight_layout()
    
    if save_fname is not None:
        fig.savefig(save_fname, transparent=False, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)
    

def visualize_3d(image_original, image_recovered, figsize=(20, 10), vmax=8, permute=[0, 1, 2], save_fname=None):
    """
    The oringal data should be (azimuthal, z, layer)
    """
    fig = plt.figure(figsize=figsize)
    axis_labels = np.array(['azimuthal', 'z', 'layer'])
    axis_scale = np.array([192, 249, 16 * 4])
    axis_labels = axis_labels[permute]
    axis_scale = axis_scale[permute]
    
    for i, (image, title) in enumerate(zip([image_original, image_recovered], ['Original', 'Decompressed'])):
        image = image.permute(*permute)
        image = image.cpu().detach().numpy()
        image = np.log2(image + 1)
    
        X, Y, Z = np.argwhere((image > 6)).T
        values = image[X, Y, Z]
        if vmax:
            values[values > vmax] = vmax

        ax = fig.add_subplot(121 + i, projection='3d')
        # Don't use normalize, just set range manually. 
        # It seems that it use up all memory. 
        # normalize = matplotlib.colors.Normalize(vmin=6, vmax=8)
        ax.scatter3D(
            X, Y, Z, 
            c=values, 
            s=1,
            cmap='Reds', 
        )
        ax.set_box_aspect(tuple(axis_scale))
        ax.set_xlabel(axis_labels[0], fontsize=20)
        ax.set_ylabel(axis_labels[1], fontsize=20)
        ax.set_zlabel(axis_labels[2], fontsize=20)
        ax.set_title(title, fontsize=25)
        
    fig.tight_layout()
    fig.subplots_adjust(wspace=.2)
    
    if save_fname is not None:
        fig.savefig(save_fname, transparent=False, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)
    
        
def plot_mse(x, y_r, y_c, transform, thresholds=np.linspace(.1, .6, 51), save_fname=None, overall_only=False):
    """
    curve of overall MSE and MSE on non-zero values for various threshold
    """
    
    X = x.cpu().detach().numpy().flatten()
    
    OA, NZ = [], []
    for t in thresholds:
        y = transform(y_r) * (y_c > t)    
        Y = y.cpu().detach().numpy().flatten()

        overall_mse = ((Y - X) ** 2).mean()
        mse = np.sum((Y - X)**2 * (X > 0)) / np.sum(X > 0)

        OA.append(overall_mse) # overall mean squared error
        NZ.append(mse) # mean squared error over non-zero ground truth values

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(thresholds, OA, label='overall')
    if overall_only == False:
        ax.plot(thresholds, NZ, label='non-zero')
    ax.legend(fontsize=16)
    
    ax.set_xlabel('threshold', fontsize=16)
    ax.set_ylabel('mse', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    plt.grid('True')
    plt.title('MSE', fontsize=18)
    fig.tight_layout()
    
    if save_fname is not None:
        fig.savefig(save_fname, transparent=False, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)
    
    return pd.DataFrame(data={'threshold': thresholds, 'overall': OA, 'nonzero': NZ})
    

if __name__ == '__main__':
    print('This is main of visualzation')
    print(help(visualize_3d))
