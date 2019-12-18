
import numpy as np
import pandas as pd
import sys
import yaml
import shutil
import uuid
import os

from store import load_artifact, store_artifact

import simulation


def calc_performance(design, sim_args):
    cl, cd = simulation.compute_coeff(design, **sim_args)
    perf = cl/cd

    if np.isnan(perf) or perf > 300:
        perf = np.nan
    return [perf, cl, cd]


def run(*, sample_name, noise_idx, sim_args, n_mod=1, mod_val=0):
    designs = load_artifact(sample_name, 'designs')
    grid_df = load_artifact(sample_name, 'grid_df')
    print(grid_df)
    grid_df_selected = grid_df[(grid_df['noise_idx'] == noise_idx)].copy()
    idx = np.arange(mod_val, len(grid_df_selected), n_mod)

    grid_df_selected = grid_df_selected.iloc[idx].copy()

    # first calculation is failing (bug), therefore do one without considering results
    calc_performance(designs[0], sim_args)


    print(len(grid_df_selected))
    print(grid_df_selected)
    
    metrics = []
    for idx, row in grid_df_selected.iterrows():
        design = designs[grid_df_selected.loc[idx, 'design_idx']]
        metrics.append(calc_performance(design, sim_args))

    metrics_dfs = pd.DataFrame(metrics, columns=['performance', 'c_lift', 'c_drag'], index=grid_df_selected.index)
    df = pd.concat([grid_df_selected, metrics_dfs], axis=1)
    store_artifact(df, sample_name, group_name='metrics_dfs', obj_name=f'{noise_idx}_{mod_val}')


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        job = yaml.load(f.read())

    tmp_folder = f'/tmp/{uuid.uuid4()}'
    current_folder = os.getcwd()

    os.mkdir(tmp_folder)
    os.chdir(tmp_folder)
    run(**job['parameter'])
    shutil.rmtree(tmp_folder)
    os.chdir(current_folder)
