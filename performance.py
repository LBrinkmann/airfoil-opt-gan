
import numpy as np
import pandas as pd

from store import load_artifact, store_artifact

import simulation


def calc_performance(design):
    perf, cl, cd = simulation.evaluate(design, return_CL_CD=True)
    return np.array([perf, cl, cd])

def run(*, sample_name, noise_idx):
    designs = load_artifact(sample_name, 'designs')
    grid_df = load_artifact(sample_name, 'grid_df')

    grid_df_selected = grid_df[grid_df['noise_idx'] == noise_idx]

    # first calculation is failing (bug), therefore do one without considering results
    calc_performance(designs[0])

    grid_df_selected[['performance', 'c_lift', 'c_drag']] = np.nan
    for idx in grid_df_selected:
        design_idx = designs[grid_df_selected.loc[idx, 'design_idx']]
        grid_df_selected.loc[idx, ['performance', 'c_lift', 'c_drag']] = calc_performance(designs[design_idx])

    store_artifact(df, sample_name, group_name='metrics_dfs', noise_idx)

