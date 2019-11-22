import argparse
import yaml
import os

EXPERIMENT_FOLDER = os.environ.get('EXPERIMENT_FOLDER', 'experiments')


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('experiment', type=str)
    args = parser.parse_args()

    model_name, mode = args.experiment.split('.')


    with open(EXPERIMENT_FOLDER + '/' + model_name + '.yml', 'r') as f:
        exp_settings =yaml.safe_load(f)
    
    if mode == 'train':
        import train
        train.run(name=model_name, **exp_settings[mode])
    elif mode == 'sample':
        import sample
        sample.run(model_name=model_name, **exp_settings[mode])
    elif mode == 'performance':
        import performance
        performance.run(model_name=model_name, **exp_settings[mode])
    elif mode == 'plot':
        import plot
        plot.run(model_name=model_name, **exp_settings[mode])
    else:
        raise ValueError('Unknown mode: ' + mode)