
import numpy as np

from store import load_artifact, store_artifact

import simulation


def calc_performance(design):
    perf, cl, cd = simulation.evaluate(design, return_CL_CD=True)
    return np.array([perf, cl, cd])

def run(*, model_name):
    designs = load_artifact(model_name, 'designs')

    design_shape = designs.shape[-2:]
    grid_shape = designs.shape[:-2]
    # print(design_shape, grid_shape)

    # for i in range(10):
    #     design = designs[-1][-1][-1][i]

    #     calc_performance(design)
    designs_flatten = np.reshape(designs, (-1, *design_shape))

    metrics = np.empty(shape=(designs_flatten.shape[0], 3))
    for i in range(designs_flatten.shape[0]):
        metrics[i] = calc_performance(designs_flatten[i])
    
    metrics = metrics.reshape((*grid_shape, 3))

    store_artifact(metrics, model_name, 'metrics')

