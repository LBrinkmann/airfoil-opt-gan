
import numpy as np
import pandas as pd
import sys
import yaml

from store import load_artifact, store_artifact

import simulation


def calc_performance(design):
    perf, cl, cd = simulation.evaluate(design, return_CL_CD=True)
    return [perf, cl, cd]

def run(*, sample_name, noise_idx):
    designs = load_artifact(sample_name, 'designs')
    grid_df = load_artifact(sample_name, 'grid_df')

    print(grid_df)
    grid_df_selected = grid_df[grid_df['noise_idx'] == noise_idx].copy()

    # first calculation is failing (bug), therefore do one without considering results
    calc_performance(designs[0])

    print(grid_df_selected)
    
    metrics = []
    for idx, row in grid_df_selected.iterrows():
        design = designs[grid_df_selected.loc[idx, 'design_idx']]
        metrics.append(calc_performance(design))

    metrics_dfs = pd.DataFrame(metrics, columns=['performance', 'c_lift', 'c_drag'], index=grid_df_selected.index)
    df = pd.concat([grid_df_selected, metrics_dfs], axis=1)
    store_artifact(df, sample_name, group_name='metrics_dfs', obj_name=str(noise_idx))


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        job = yaml.load(f.read())
    run(**job['parameter'])