from os.path import basename, splitext
from pathlib import Path

import wandb

if __name__ == '__main__':
    models = list((Path() / 'learned_models' / '03_05' / 'autoencoder').glob('*50epochs*'))

    run = wandb.init(project='diplom_autoencoders', entity='ars860')
    run.name = 'Models'

    for model in models:
        artifact = wandb.Artifact(splitext(basename(str(model)))[0], type='model')
        artifact.add_file(str(model))
        run.log_artifact(artifact)

    # print(models)