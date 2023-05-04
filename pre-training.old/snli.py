from typing import Any, Optional, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import lightning.pytorch as pl
import transformers
import os
import sys
import gc

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from rich.progress import track
from rich.table import Table
from rich.console import Console
from rich import print as rprint


from datalmodule import SNLIDataset, LitDataLoader
from models import LitRoBERTaNLI

def view_data(data: "pd.DataFrame", n_rows: Optional[int]=5, use_ruler: Optional[bool]=False, title: Optional[str]="title") -> None:
    """
    This function prints an overview of the data.

    :param data: the dataframe
    :param n_rows: number of rows that you want to view.
    :param use_ruler: whether to add any seperator after printing the data.
    :param title: a title for the dataframe.

    :returns: None
    """

    caption = f"This data originally has {data.shape[0]} records"
    data = data.head(n_rows)

    table = Table(title=title, caption=caption)
    
    for col in data.columns:
        table.add_column(col, justify="left", header_style="#F2E863")
    
    for row in data.iterrows():
        table.add_row(*(str(val) for val in row[1]))

    console = Console()
    
    if use_ruler:
        console.rule(f"[bold #F25757]{title}[/bold #F25757]")

    console.print(table)



if __name__ == "__main__":

    df_train = pd.read_csv("../data/snli/archive/snli_1.0_train.csv")
    df_val = pd.read_csv("../data/snli/archive/snli_1.0_dev.csv")

    df_train = df_train[['sentence1', 'sentence2', 'gold_label']]
    df_val = df_val[['sentence1', 'sentence2', 'gold_label']]

    df_train = df_train.dropna()
    df_val = df_val.dropna()

    df_train = df_train.sample(n=200_000, random_state=42)

    df_train = df_train.loc[df_train["gold_label"] != "-"]
    df_val = df_val.loc[df_val["gold_label"] != "-"]

    view_data(df_train, use_ruler=True, title="Training Dataset")
    view_data(df_val, use_ruler=True, title="Validation Dataset")    


    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = SNLIDataset(df_train, tokenizer, max_length=512, padding='max_length',
                                truncation=True, return_attention_mask=True, add_special_tokens=True,
                                return_tensors="pt")
    
    val_dataset = SNLIDataset(df_val, tokenizer, max_length=512, padding='max_length',
                                truncation=True, return_attention_mask=True, add_special_tokens=True,
                                return_tensors="pt")
    

    train_dl = LitDataLoader(train_dataset, batch_size=32).train_dataloader()
    val_dl = LitDataLoader(val_dataset, batch_size=128).val_dataloader()

    model = LitRoBERTaNLI(3)

    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")

    early_stop = pl.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
    model_check = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min",
                                               dirpath="./checkpoints", filename="robertaSNLI-best-checkpoint",
                                               save_top_k=1, verbose=True, save_on_train_epoch_end=False)
    
    prog_bar = pl.callbacks.RichProgressBar()

    wandb_logger = WandbLogger(name="snli_metrics", project="SNLI")

    torch.cuda.empty_cache()
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         precision=16,
                         max_epochs=100,
                         strategy="deepspeed_stage_3",
                         logger=wandb_logger,
                         callbacks=[early_stop, model_check, prog_bar],
                         accumulate_grad_batches=10)
    
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    gc.collect()


