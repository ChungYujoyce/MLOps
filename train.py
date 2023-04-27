import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data import DataModule
from model import ColaModel
import wandb
from pytorch_lightning.loggers import WandbLogger

import hydra
from omegaconf.omegaconf import OmegaConf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", filename="best-checkpoint", monitor="valid/loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )
    
    wandb_logger = WandbLogger(project="MLOps", entity="joyce-1010")
    
    trainer = pl.Trainer(
        gpus = (1 if torch.cuda.is_available() else 0),
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()

if __name__ == "__main__":
    main()