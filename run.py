import train
import argparse
import yaml


MODEL_FOLDER = os.environ.get('MODEL_FOLDER', 'trained_gan')
DATA_FOLDER = os.environ.get('DATA_FOLDER')
LOG_FOLDER = os.environ.get('LOG_FOLDER')
EXPERIMENT_FOLDER = os.environ.get('EXPERIMENT_FOLDER', 'experiments')


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    args = parser.parse_args()


    with open(EXPERIMENT_FOLDER + '/' + args.name + '.yml', 'r') as f:
        exp_settings =yaml.safe_load(f)
    
    if settings['mode'] == 'train':
        train.run(name=args.name, **exp_settings)

    