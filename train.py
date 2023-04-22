import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data import DataModule
from model import ColaModel
import wandb
from pytorch_lightning.loggers import WandbLogger


def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", filename="best-checkpoint", monitor="valid/loss", mode="min"
    )
    earlt_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps", entity="joyce-1010")
    trainer = pl.Trainer(
        default_root_dir = "logs",
        gpus = (1 if torch.cuda.is_available() else 0),
        max_epochs = 10, 
        fast_dev_run = False,
        deterministic=True,
        logger = wandb_logger,
        log_every_n_steps=10,
        callbacks = [checkpoint_callback, earlt_stopping_callback]
    )
    trainer.fit(cola_model, cola_data)
    

if __name__ == "__main__":
    main()

