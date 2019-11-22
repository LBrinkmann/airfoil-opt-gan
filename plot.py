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

    grid_shape = designs.shape[:-2]

    if len(grid_shape) == 4:
        dim_labels = [
            ['{:.2f}'.format(d) for d in latend_grid[:,0,0,1]],
            ['dim 2: {:.2f}'.format(d) for d in latend_grid[0,:,0,0]],
            ['dim 3: {:.2f}'.format(d) for d in latend_grid[0,0,:,2]],
        ]

        x_lim = [np.min(designs[:,:,:,:,:,0]), np.max(designs[:,:,:,:,:,0])]
        y_lim = [np.min(designs[:,:,:,:,:,1]), np.max(designs[:,:,:,:,:,1])]

        plot_3Dgrid(designs, dim_labels, 'designs', model_name, plot_args, x_lim, y_lim)

    else:
        dim_labels = [
            ['dim 1: {:.2f}'.format(d) for d in latend_grid[:,0,0]],
            ['dim 2: {:.2f}'.format(d) for d in latend_grid[0,:,1]],
        ]     
        plot_2Dgrid(designs, dim_labels, 'designs', model_name, plot_args)


def plot_designs(designs, ax, plot_args, x_lim, y_lim):
    for i in range(designs.shape[0]):
        ax.plot(designs[i,:,0], designs[i,:,1], **plot_args)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.axis('off')
    ax.axis('equal')


def plot_2Dgrid(designs, dim_labels, plot_name, model_name, plot_args, x_lim, y_lim):
    # 2D Plot
    fig = plt.figure(figsize=(8, 8))

    design_shape = designs.shape[-2:]
    grid_shape = designs.shape[:-2]

    ax = fig.subplots(grid_shape[0], grid_shape[1])

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            plot_designs(designs[i,j], ax[i,j], plot_args, x_lim, y_lim)

    # # TODO: double check of labels are correctly assigned
    # for _ax, l in zip(ax[0, :], dim_labels[1]):
    #     _ax.set_title(l)

    # for _ax, l in zip(ax[:, 0], dim_labels[0]):
    #     _ax.set_ylabel(l)


    fig.suptitle(plot_name)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plot_path = get_artifact_path(model_name, plot_name + '.png', group_name='plots')

    plt.savefig(plot_path, dpi=600)
    plt.close()


def plot_3Dgrid(designs, dim_labels, plot_name, model_name, plot_args, x_lim, y_lim):
    for i, d in enumerate(dim_labels[0]):
        plot_2Dgrid(designs[i], dim_labels[1:], plot_name + '_' + d, model_name, plot_args, x_lim, y_lim)


