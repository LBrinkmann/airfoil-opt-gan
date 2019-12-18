"""
Plots samples or new shapes in the semantic space.

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
"""
import sys
import yaml
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from utils import gen_grid


import numpy as np

from store import load_artifact, store_artifact, get_artifact_path, load_artifacts


def run(*, sample_name, plot_args, noise_idx=None):
    designs = load_artifact(sample_name, 'designs')
    metrics_dfs = load_artifacts(sample_name, 'metrics_dfs')

    print('Loaded dataframes')

    metrics_df = pd.concat(metrics_dfs)

    print(metrics_df)

    if noise_idx is not None:
        metrics_df = metrics_df[metrics_df['noise_idx'] == noise_idx]

    print(metrics_df)

    print('Merged dataframes')

    points_per_dim = metrics_df['points_per_dim'].max()
    latent_dim = metrics_df['latent_dim'].max()

    dim_idx_cols = [f'dim_{i}_idx' for i in range(latent_dim)]
    dim_val_cols = [f'dim_{i}_value' for i in range(latent_dim)]

    metrics = ['performance', 'c_lift', 'c_drag']

    # TODO: this is just a quick fix
    metrics_df.loc[metrics_df['performance'] < -1000, 'performance'] = np.nan

    grid_val_agg = metrics_df.groupby(dim_idx_cols)[dim_val_cols].min()
    metrics_df_agg = metrics_df.groupby(dim_idx_cols)[metrics].mean()
    design_idx_agg = metrics_df.groupby(dim_idx_cols)['design_idx'].apply(list)

    df = pd.concat([grid_val_agg, metrics_df_agg, design_idx_agg], axis=1).reset_index()

    df['norm_performance'] = (
        df['performance']-df['performance'].min())/(df['performance'].max()-df['performance'].min())

    x_lim = [np.min(designs[:,0]), np.max(designs[:,0])]
    y_lim = [np.min(designs[:,1]), np.max(designs[:,1])]

    subplot_args = {'design_x_lim': x_lim, 'design_y_lim': y_lim}


    if latent_dim == 2:
        x_dim = 0
        y_dim = 1
        title = 'designs'
        if noise_idx is not None:
            title += f'_{noise_idx}'
        plot_2Dgrid(df, designs, points_per_dim, x_dim, y_dim, sample_name, title, plot_args, subplot_args)
    else:
        x_dim = 2
        y_dim = 3
        add_dim_value = [f'dim_{i}_value'for i in range(0, latent_dim - 2)]
        add_dim_idx = [f'dim_{i}_idx'for i in range(0, latent_dim - 2)]
        for idx, dfg in df.groupby(add_dim_idx):
            title = ' - '.join(f'{v:.2f}' for v in dfg.iloc[0][add_dim_value])
            print(f'Plot {title}')
            plot_2Dgrid(dfg, designs, points_per_dim, x_dim, y_dim, sample_name, title, plot_args, subplot_args)


def plot_designs(designs, metric, ax, plot_args, subplot_args):
    for i in range(designs.shape[0]):
        ax.plot(designs[i,:,0], designs[i,:,1], **plot_args)

    cmap = cm.get_cmap('viridis')
    color = cmap(metric)
    ax.set_xlim(*subplot_args['design_x_lim'])
    ax.set_ylim(*subplot_args['design_y_lim'])
   
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    # ax.axis('off')
    ax.set_facecolor(color)
    ax.axis('equal')

def plot_2Dgrid(df, designs, points_per_dim, x_dim, y_dim, sample_name, plot_name, plot_args, subplot_args):
    # 2D Plot
    fig = plt.figure(figsize=(8, 8))

    design_shape = designs.shape[-2:]
    grid_shape = designs.shape[:-2]

    ax = fig.subplots(points_per_dim, points_per_dim)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)

    for idx, row in df.iterrows():
        metric = row['norm_performance']
        design_idx = row['design_idx']
        _ax = ax[row[f'dim_{x_dim}_idx'], row[f'dim_{y_dim}_idx']]
        plot_designs(designs[design_idx], metric, _ax, plot_args, subplot_args)

    fig.suptitle(plot_name)

    plot_path = get_artifact_path(sample_name, plot_name + '.png', group_name='plots')

    plt.savefig(plot_path, dpi=600)
    plt.close()


# def plot_3Dgrid(designs, metrics, dim_labels, plot_name, sample_name, plot_args, subplot_args):
#     for i, d in enumerate(dim_labels[0]):
#         plot_2Dgrid(designs[i], metrics[i], dim_labels[1:], plot_name + '_' + d, sample_name, plot_args, subplot_args)



if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        job = yaml.load(f.read())
    run(**job['parameter'])