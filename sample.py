
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


def create_lines(dimensions, boundaries, grid_points, line_points):
    all_grid = []
    for m_dim in range(dimensions):
        points_per_dim = [grid_points] * m_dim + [line_points] + [grid_points] * (dimensions - m_dim - 1)
        axis = (np.linspace(boundaries[0], boundaries[1], ppd) for ppd in points_per_dim)
        grid = np.meshgrid(*axis)
        grid = np.stack((g.reshape(-1) for g in grid), axis=-1)
        all_grid.append(grid)
    all_grid = np.concatenate(all_grid, axis=0)
    return all_grid, np.arange(len(all_grid)).reshape((-1, 1))



def add_noise_dimension(latend, grid_idx):
    noise = np.random.normal(scale=0.5, size=(noise_sample, model.noise_dim))
    noise = np.tile(noise, (np.prod(latend.shape[:-1]),1))

    latend = np.expand_dims(latend, axis=-2)
    latend = np.broadcast_to(latend, (latend.shape[0], noise_sample, latend.shape[-1]))  
    latend_flat = latend.reshape((-1, latend.shape[-1]))


def add_noise(latend, grid_idx, noise_sample, noise_dim):
    # each latend grid point gets the same set of noise vectors
    noise = np.random.normal(scale=0.5, size=(noise_sample, noise_dim))
    noise_idx = np.arange(noise_sample)
    noise = np.tile(noise, (latend.shape[0],1))
    noise_idx = np.tile(noise_idx, latend.shape[0])

    # repeat each latend grid vector "noise_sample" times
    latend = np.repeat(latend, noise_sample, axis=0)
    grid_idx = np.repeat(grid_idx, noise_sample, axis=0)

    return latend, grid_idx, noise, noise_idx


def run(model_name, sample_name, latend_bounds, points_per_dim, noise_sample, mesh_type, line_points=None):

    model_path = get_model_path(model_name)

    model = GAN.restore(model_path)

    if mesh_type == 'grid':
        latend, grid_idx = create_grid(model.latent_dim, latend_bounds, points_per_dim)
        latend, grid_idx, noise, noise_idx = add_noise(latend, grid_idx, noise_sample, model.noise_dim)
        design_idx = np.arange(len(designs))
        grid_df = pd.concat(
            [
                pd.DataFrame(design_idx, columns=['design_idx']),
                pd.DataFrame(latend, columns=[f'dim_{d}_value' for d in range(latend.shape[1])]),
                pd.DataFrame(grid_idx, columns=[f'dim_{d}_idx' for d in range(grid_idx.shape[1])]),
                pd.DataFrame(noise_idx, columns=['noise_idx']),
            ],
            axis=1
        )
        grid_df['points_per_dim'] = points_per_dim
        grid_df['noise_sample'] = noise_sample
    elif mesh_type == 'line':
        latend, line_idx = create_lines(model.latent_dim, latend_bounds, grid_points=points_per_dim, line_points=30)
        latend, line_idx, noise, noise_idx = add_noise(latend, line_idx, noise_sample, model.noise_dim)
        design_idx = np.arange(len(latend))
        grid_df = pd.concat(
            [
                pd.DataFrame(design_idx, columns=['design_idx']),
                pd.DataFrame(latend, columns=[f'dim_{d}_value' for d in range(latend.shape[1])]),
                pd.DataFrame(line_idx, columns=['line_idx']),
                pd.DataFrame(noise_idx, columns=['noise_idx']),
            ],
            axis=1
        )
        grid_df['points_per_dim'] = points_per_dim
        grid_df['noise_sample'] = noise_sample


    d_list = []
    batch_size = 1000
    for i in range(0, len(latend), batch_size):
        designs = model.synthesize(latend[i:i+batch_size], noise=noise[i:i+batch_size])
        d_list.append(designs)
    designs = np.concatenate(d_list)

    store_artifact(designs, sample_name, 'designs')
    store_artifact(noise, sample_name, 'noise')

    grid_df['latent_dim'] = model.latent_dim

    store_artifact(grid_df, sample_name, 'grid_df')
    print(grid_df)


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        job = yaml.load(f.read())
    run(**job['parameter'])
