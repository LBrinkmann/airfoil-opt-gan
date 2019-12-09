"""
Plots samples or new shapes in the semantic space.

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
"""

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from utils import gen_grid


import numpy as np

from store import load_artifact, store_artifact, get_artifact_path


def run(*, model_name, plot_args):
    designs = load_artifact(model_name, 'designs')
    latend_grid = load_artifact(model_name, 'latend_grid')
    metrics = load_artifact(model_name, 'metrics')

    # import ipdb; ipdb.set_trace()

    grid_shape = designs.shape[:-2]

    metrics[metrics==-np.inf] = np.nan
    metrics[metrics==np.inf] = np.nan


    if len(grid_shape) == 4:
        dim_labels = [
            ['{:.2f}'.format(d) for d in latend_grid[:,0,0,1]],
            ['dim 2: {:.2f}'.format(d) for d in latend_grid[0,:,0,0]],
            ['dim 3: {:.2f}'.format(d) for d in latend_grid[0,0,:,2]],
        ]

        x_lim = [np.min(designs[:,:,:,:,:,0]), np.max(designs[:,:,:,:,:,0])]
        y_lim = [np.min(designs[:,:,:,:,:,1]), np.max(designs[:,:,:,:,:,1])]

        median_metric = np.nanmean(metrics[:,:,:,:,0], axis=-1)
        maxm = np.nanmax(median_metric)
        minm = np.nanmin(median_metric)
        norm_metric = (median_metric - minm) / (maxm - minm)

        # perf_lim = [0, 120]

        subplot_args = {'design_x_lim': x_lim, 'design_y_lim': y_lim}

        plot_3Dgrid(designs, norm_metric, dim_labels, 'designs', model_name, plot_args, subplot_args)

    else:
        x_lim = [np.min(designs[:,:,:,:,0]), np.max(designs[:,:,:,:,0])]
        y_lim = [np.min(designs[:,:,:,:,1]), np.max(designs[:,:,:,:,1])]

        median_metric = np.nanmedian(metrics[:,:,:,0], axis=-1)
        maxm = np.nanmax(median_metric)
        minm = np.nanmin(median_metric)
        norm_metric = (median_metric - minm) / (maxm - minm)

        subplot_args = {'design_x_lim': x_lim, 'design_y_lim': y_lim}

        dim_labels = [
            ['dim 1: {:.2f}'.format(d) for d in latend_grid[:,0,0]],
            ['dim 2: {:.2f}'.format(d) for d in latend_grid[0,:,1]],
        ]     
        plot_2Dgrid(designs, norm_metric, dim_labels, 'designs', model_name, plot_args, subplot_args)


def plot_designs(designs, metrics, ax, plot_args, subplot_args):
    for i in range(designs.shape[0]):
        ax.plot(designs[i,:,0], designs[i,:,1], **plot_args)

    cmap = cm.get_cmap('viridis')
    color = cmap(metrics)
    ax.set_xlim(*subplot_args['design_x_lim'])
    ax.set_ylim(*subplot_args['design_y_lim'])
   
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    # ax.axis('off')
    ax.set_facecolor(color)
    ax.axis('equal')


def plot_2Dgrid(designs, metrics, dim_labels, plot_name, model_name, plot_args, subplot_args):
    # 2D Plot
    fig = plt.figure(figsize=(8, 8))

    design_shape = designs.shape[-2:]
    grid_shape = designs.shape[:-2]

    ax = fig.subplots(grid_shape[0], grid_shape[1])

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)


    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            plot_designs(designs[i,j], metrics[i,j], ax[i,j], plot_args, subplot_args)

    # # TODO: double check of labels are correctly assigned
    # for _ax, l in zip(ax[0, :], dim_labels[1]):
    #     _ax.set_title(l)

    # for _ax, l in zip(ax[:, 0], dim_labels[0]):
    #     _ax.set_ylabel(l)


    fig.suptitle(plot_name)
    # plt.xticks([], [])
    # plt.yticks([], [])

    # plt.tick_params(
    #     axis='both',
    #     which='both',
    #     bottom=False,
    #     left=False,
    #     right=False,
    #     top=False,
    #     labelbottom=False
    # )
    # plt.tight_layout()

    plot_path = get_artifact_path(model_name, plot_name + '.png', group_name='plots')

    plt.savefig(plot_path, dpi=600)
    plt.close()


def plot_3Dgrid(designs, metrics, dim_labels, plot_name, model_name, plot_args, subplot_args):
    for i, d in enumerate(dim_labels[0]):
        plot_2Dgrid(designs[i], metrics[i], dim_labels[1:], plot_name + '_' + d, model_name, plot_args, subplot_args)


