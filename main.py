from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything, Trainer

import argparse
import json
import os

from net import LightningVAE
from data import MNISTDataModule


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="File with model configuration")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        try:
            params = json.load(f)
        except json.decoder.JSONDecodeError as error:
            print(error)

    logger = TensorBoardLogger(save_dir=params["logger"]["save_dir"],
                               name="VAE")
    seed_everything(params['seed'])

    experiment = LightningVAE(params)
    data = MNISTDataModule(data_dir=params["data"]["save_dir"],
                           batch_size=params["data"]["batch_size"],
                           download=params["data"]["download"])
    data.setup()
    trainer = Trainer(logger=logger,
                      callbacks=[
                          ModelCheckpoint(save_top_k=1,
                                          dirpath=os.path.join(logger.log_dir, "checkpoints"),
                                          monitor="val_loss",
                                          save_last=True),
                      ], **params["trainer"])

    os.makedirs(os.path.join(logger.log_dir, "Reconstructions"), exist_ok=True)
    os.makedirs(os.path.join(logger.log_dir, "Samples"), exist_ok=True)

    trainer.fit(experiment, datamodule=data)


if __name__ == "__main__":
    run()
