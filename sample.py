
import argparse
import os
import numpy as np
import pandas as pd
import yaml
import sys

from store import store_artifact, get_model_path
from gan import GAN


def create_grid(dimensions, boundaries, points_per_dim):
    # create a grid of latend values
    axis = (np.linspace(boundaries[0], boundaries[1], points_per_dim) for i in range(dimensions))
    grid = np.meshgrid(*axis)
    grid = np.stack((g.reshape(-1) for g in grid), axis=-1)

    # create a corresponding grid of idx 
    axis_idx = (np.arange(points_per_dim) for i in range(dimensions))
    grid_idx = np.meshgrid(*axis_idx)
    grid_idx = np.stack((g.reshape(-1) for g in grid_idx), axis=-1)

    return grid, grid_idx


def add_noise_dimension(latend_grid, grid_idx):
    noise = np.random.normal(scale=0.5, size=(noise_sample, model.noise_dim))
    noise = np.tile(noise, (np.prod(latend_grid.shape[:-1]),1))

    latend_grid = np.expand_dims(latend_grid, axis=-2)
    latend_grid = np.broadcast_to(latend_grid, (latend_grid.shape[0], noise_sample, latend_grid.shape[-1]))  
    latend_grid_flat = latend_grid.reshape((-1, latend_grid.shape[-1]))


def add_noise(latend_grid, grid_idx, noise_sample, noise_dim):
    # each latend grid point gets the same set of noise vectors
    noise = np.random.normal(scale=0.5, size=(noise_sample, noise_dim))
    noise_idx = np.arange(noise_sample)
    noise = np.tile(noise, (latend_grid.shape[0],1))
    noise_idx = np.tile(noise_idx, latend_grid.shape[0])

    # repeat each latend grid vector "noise_sample" times
    latend_grid = np.repeat(latend_grid, noise_sample, axis=0)
    grid_idx = np.repeat(grid_idx, noise_sample, axis=0)

    return latend_grid, grid_idx, noise, noise_idx


def run(model_name, sample_name, latend_bounds, points_per_dim, noise_sample):

    model_path = get_model_path(model_name)

    model = GAN.restore(model_path)

    latend_grid, grid_idx = create_grid(model.latent_dim, latend_bounds, points_per_dim)

    latend_grid, grid_idx, noise, noise_idx = add_noise(latend_grid, grid_idx, noise_sample, model.noise_dim)

    designs = model.synthesize(latend_grid, noise=noise)
    design_idx = np.arange(len(designs))

    store_artifact(designs, sample_name, 'designs')
    store_artifact(noise, sample_name, 'noise')


    grid_df = pd.concat(
        [
            pd.DataFrame(design_idx, columns=['design_idx']),
            pd.DataFrame(latend_grid, columns=[f'dim_{d}_value' for d in latend_grid.shape[1]]),
            pd.DataFrame(grid_idx, columns=[f'dim_{d}_idx' for d in grid_idx.shape[1]]),
            pd.DataFrame(noise_idxs, columns=['noise_idx']),
        ],
        axis=1
    )
    store_artifact(grid_df, sample_name, 'grid_df')
    print(grid_df)


if __name__ == '__main__':
    print(sys.argv[1])
    with open(sys.argv[1], 'r') as f:
        job = yaml.load(f.read())
    run(**job['parameter'])