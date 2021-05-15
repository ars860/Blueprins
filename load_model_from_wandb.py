import argparse

import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)

    args = parser.parse_args()

    if args.name is None:
        raise ValueError('name should not be None')

    run = wandb.init()
    artifact = run.use_artifact(f'diplom_segmentation/{args.name}')
    artifact_dir = artifact.download()

    print(f'model downloaded into {artifact_dir}')
