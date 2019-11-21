
"""
Trains an Hmodel, and visulizes results

Author(s): Wei Chen (wchen459@umd.edu)
"""

import argparse
import os
import numpy as np
import yaml
from importlib import import_module

from gan import GAN
from manifold_evaluation.diversity import ci_rdiv
from manifold_evaluation.likelihood import ci_mll
from manifold_evaluation.consistency import ci_cons
from manifold_evaluation.smoothness import ci_rsmth
from utils import ElapsedTimer

MODEL_FOLDER = os.environ.get('MODEL_FOLDER', 'trained_gan')
DATA_FOLDER = os.environ.get('DATA_FOLDER')
LOG_FOLDER = os.environ.get('LOG_FOLDER')
EXPERIMENT_FOLDER = os.environ.get('EXPERIMENT_FOLDER', 'experiments')


def run(*, name, mode, save_interval, latent_dim, noise_dim, bezier_degree, train_steps, batch_size, symm_axis, bounds, plotting):

    # Read dataset
    data_fname = 'airfoil_interp.npy'
    X = np.load(data_fname)

    if plotting:
        from shape_plot import plot_samples
        print('Plotting training samples ...')
        samples = X[np.random.choice(list(range(X.shape[0])), size=36)]

        plot_samples(None, samples, scale=1.0, scatter=False,
                     symm_axis=symm_axis, lw=1.2, alpha=.7, c='k', fname='samples')

    # Split training and test data
    test_split = 0.8
    N = X.shape[0]
    split = int(N*test_split)
    X_train = X[:split]
    X_test = X[split:]

    model_path = MODEL_FOLDER + '/' + name
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Train
   
    if mode == 'startover':
        model = GAN(latent_dim, noise_dim, X_train.shape[1], bezier_degree, bounds)
        timer = ElapsedTimer()

        model.train(X_train, batch_size=batch_size,
                    train_steps=train_steps, save_interval=save_interval, model_path=model_path)
        elapsed_time = timer.elapsed_time()
        runtime_mesg = 'Wall clock time for training: %s' % elapsed_time
        print(runtime_mesg)
        # runtime_file = open('gan/runtime.txt', 'w')
        # runtime_file.write('%s\n' % runtime_mesg)
        # runtime_file.close()
    else:
        model = GAN.restore(model_path)

    if plotting:
        from shape_plot2 import plot_grid
        print('Plotting synthesized shapes ...')
        plot_grid(5, gen_func=model.synthesize, d=latent_dim, bounds=bounds, scale=1.0, scatter=False, symm_axis=symm_axis,
                  alpha=.7, lw=0.2, fname='gan/synthesized')


    n_runs = 10

    mll_mean, mll_err = ci_mll(n_runs, model.synthesize, X_test)
    rdiv_mean, rdiv_err = ci_rdiv(n_runs, X, model.synthesize)
    cons_mean, cons_err = ci_cons(
        n_runs, model.synthesize, latent_dim, bounds)  # Only for GANs
    rsmth_mean, rsmth_err = ci_rsmth(n_runs, model.synthesize, X_test)

    results_mesg_1 = 'Mean log likelihood: %.1f +/- %.1f' % (mll_mean, mll_err)
    results_mesg_2 = 'Relative diversity: %.3f +/- %.3f' % (
        rdiv_mean, rdiv_err)
    results_mesg_3 = 'Consistency: %.3f +/- %.3f' % (cons_mean, cons_err)
    results_mesg_4 = 'Smoothness: %.3f +/- %.3f' % (rsmth_mean, rsmth_err)

    results_file = open('gan/results.txt', 'w')

    print(results_mesg_1)
    results_file.write('%s\n' % results_mesg_1)
    print(results_mesg_2)
    results_file.write('%s\n' % results_mesg_2)
    print(results_mesg_3)
    results_file.write('%s\n' % results_mesg_3)
    print(results_mesg_4)
    results_file.write('%s\n' % results_mesg_4)

    rdiv_means = []
    rdiv_errs = []
    for k in range(latent_dim):
        rdiv_mean_k, rdiv_err_k = ci_rdiv(
            100, X, model.synthesize, latent_dim, k, plot_bounds)
        rdiv_means.append(rdiv_mean_k)
        rdiv_errs.append(rdiv_err_k)
        results_mesg_k = 'Relative diversity for latent dimension %d: %.3f +/- %.3f' % (
            k, rdiv_mean_k, rdiv_err_k)
        print(results_mesg_k)
        results_file.write('%s\n' % results_mesg_k)

    results_file.close()

    # if plotting:
    #     import matplotlib.pyplot as plt

    #     plt.figure()
    #     plt.errorbar(np.arange(latent_dim)+1, rdiv_means, yerr=rdiv_err_k)
    #     plt.xlabel('Latent Dimensions')
    #     plt.ylabel('Relative diversity')
    #     plt.savefig('rdiv.svg', dpi=600)

    print('All completed :)')


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    args = parser.parse_args()


    with open(EXPERIMENT_FOLDER + '/' + args.name + '.yml', 'r') as f:
        exp_settings =yaml.safe_load(f)

    run(name=args.name, **exp_settings)
