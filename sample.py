
import argparse
import os
import numpy as np
import yaml

from store import store_artifact, get_model_path
from gan import GAN


def create_grid(dimensions, boundaries, points_per_dim):
    axis = (np.linspace(boundaries[0], boundaries[1], points_per_dim) for i in range(dimensions))
    grid = np.meshgrid(*axis)
    grid = np.stack(grid, axis=-1)
    return grid


def gen_designs(latend_grid, model, noise_sample):
    # repeat grid along dimension to have multiple samples per grid point
    latend_grid = np.expand_dims(latend_grid, axis=-2)
    latend_grid = np.broadcast_to(latend_grid, (*latend_grid.shape[:-2], noise_sample, latend_grid.shape[-1]))  

    latend_grid_flat = latend_grid.reshape((-1, latend_grid.shape[-1]))
    noise = np.random.normal(scale=0.5, size=(latend_grid_flat.shape[0], model.noise_dim))

    X = model.synthesize(latend_grid_flat, noise=noise)
    X = X.reshape((*latend_grid.shape[:-1], -1, 2))
    return X



def run(*, model_name, latend_bounds, points_per_dim, noise_sample):

    model_path = get_model_path(model_name)

    model = GAN.restore(model_path)

    latend_grid = create_grid(model.latent_dim, latend_bounds, points_per_dim)

    designs = gen_designs(latend_grid, model, noise_sample)

    print(designs.shape)
    print(latend_grid.shape)

    store_artifact(designs, model_name, 'designs')
    store_artifact(latend_grid, model_name, 'latend_grid')
