from typing import Any, Optional, List, Tuple, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import torchmetrics
import lightning.pytorch as pl

from deepspeed.ops.adam import FusedAdam

from lightning.pytorch.utilities import rank_zero_only

pl.seed_everything(42)
torch.set_float32_matmul_precision('medium')

class LitRoBERTaNLI(pl.LightningModule):
    def __init__(self, n_classes: int, lr: Optional[float]=1e-4, min_lr: Optional[float]=1e-6, 
                 T0: Optional[int]=100, weight_decay=1e-2):
        
        super().__init__()
        
        self.n_classes = n_classes
        self.lr = lr
        self.min_lr = min_lr
        self.T0 = T0
        self.weight_decay = weight_decay

        if n_classes == 1:
            self.train_metric_collection = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(task="binary"),
                torchmetrics.F1Score(task="binary"),
                torchmetrics.Recall(task="binary")
            ])

            self.val_metric_collection = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(task="binary"),
                torchmetrics.F1Score(task="binary"),
                torchmetrics.Recall(task="binary")
            ])

            
        
        else:
            self.train_metric_collection = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(task="multiclass", num_classes=n_classes),
                torchmetrics.F1Score(task="multiclass", num_classes=n_classes),
                torchmetrics.Recall(task="multiclass", num_classes=n_classes)
            ])

            self.val_metric_collection = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(task="multiclass", num_classes=n_classes),
                torchmetrics.F1Score(task="multiclass", num_classes=n_classes),
                torchmetrics.Recall(task="multiclass", num_classes=n_classes)
            ])
            

        self.roberta_model = transformers.RobertaModel.from_pretrained('roberta-base', return_dict=True)
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, x: "torch.Tensor", attn_mask: "torch.Tensor") -> "torch.Tensor":
        pooler_out = self.roberta_model(x, attention_mask=attn_mask)["pooler_output"]
        out = self.classifier(pooler_out)

        return out
    
    def compute_loss(self, logits: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        
        if self.n_classes == 1:
            y = y.reshape(-1, 1)
            loss = F.binary_cross_entropy_with_logits(logits, y.type_as(logits))
            return loss
        
        loss = F.cross_entropy(logits, y)
        return loss
    

    def _common_steps(self, batch: "torch.Tensor", batch_idx: "torch.Tensor"):
        input_ids, attn_mask, y = batch["input_ids"], batch["attention_mask"], batch["label"]
        logits = self(input_ids, attn_mask)
        loss = self.compute_loss(logits, y)

        return loss, logits



    def training_step(self, batch: "torch.Tensor", batch_idx: "torch.Tensor"):

        loss, logits = self._common_steps(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, logger=True, sync_dist=True)

        if self.n_classes == 1:
            y = batch["label"].reshape(-1, 1)
            metric = self.train_metric_collection(logits, y)

            self.log('train_acc', metric['BinaryAccuracy'], prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_f1', metric['BinaryF1Score'], prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_recall', metric['BinaryRecall'], prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)


        else:
            y = batch["label"]
            logits = torch.argmax(logits, dim=1)
            metric = self.train_metric_collection(logits, y)

            self.log('train_acc', metric['MulticlassAccuracy'], prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_f1', metric['MulticlassF1Score'], prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_recall', metric['MulticlassRecall'], prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            


        return {"loss": loss}
    
    def validation_step(self, batch: "torch.Tensor", batch_idx: "torch.Tensor"):

        loss, logits = self._common_steps(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        if self.n_classes == 1:
            y = batch["label"].reshape(-1, 1)
            metric = self.val_metric_collection(logits, y)

            self.log('val_acc', metric['BinaryAccuracy'], prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_f1', metric['BinaryF1Score'], prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_recall', metric['BinaryRecall'], prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)


        else:
            y = batch["label"]
            logits = torch.argmax(logits, dim=1)

            metric = self.val_metric_collection(logits, y)

            self.log('val_acc', metric['MulticlassAccuracy'], prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_f1', metric['MulticlassF1Score'], prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_recall', metric['MulticlassRecall'], prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return {"val_loss":loss} 
    
    def configure_optimizers(self) -> Any:
        
        optimizer = FusedAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.T0, eta_min=self.min_lr)

        return {
            "optimizer" : optimizer,
            "lr_scheduler" : scheduler
        }
    

    










