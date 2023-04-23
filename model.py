import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import torchmetrics
import wandb
import pandas as pd
wandb.init(project="mlops")

from omegaconf import OmegaConf, DictConfig
import hydra


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=3e-5):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch["input_ids"], batch["attention_mask"], labels=batch["label"])
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        sentences = batch["sentence"]
        
        labels = batch["label"]
        outputs = self.forward(batch["input_ids"], batch["attention_mask"], labels=batch["label"])
        preds = torch.argmax(outputs.logits, 1)
        
        val_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)
        
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", val_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": outputs.logits, "sentences": sentences}
    
    def validation_epoch_end(self, outputs):
        sentences = [s for x in outputs for s in x["sentences"]]
        
        labels = torch.cat([x["labels"] for x in outputs]).cpu()
        logits = torch.cat([x["logits"] for x in outputs]).cpu()
        preds = torch.argmax(logits, 1)
        
        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )
        wrong_df = df[df["Label"] != df["Predicted"]]
        
        self.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": self.global_step,
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.numpy(), y_true=labels.numpy()
                )
            }
        )

        return 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
