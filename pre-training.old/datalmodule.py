from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import torch
import torch.utils.data as td
import lightning.pytorch as pl
import transformers


class SNLIDataset(td.Dataset):
    def __init__(self, data: "pd.DataFrame", tokenizer: "transformers.AutoTokenizer", **kwargs):
        
        self.data = data
        self.tokenizer = tokenizer
        self.kwargs = kwargs

        self.LABEL2IDX = {"neutral": 0, "entailment": 1, "contradiction": 2}

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx: int): 

        s1 = self.data.iloc[idx]['sentence1']
        s2 = self.data.iloc[idx]['sentence2']
        label = self.data.iloc[idx]['gold_label']

        tokenize_res = self.tokenizer(s1, s2, **self.kwargs)

        input_ids = tokenize_res["input_ids"]
        attn_mask = tokenize_res["attention_mask"]

        return {
            "input_ids" : input_ids.to(dtype=torch.long).flatten(),
            "attention_mask" :  attn_mask.to(dtype=torch.long).flatten(),
            "label" : torch.tensor(self.LABEL2IDX[label], dtype=torch.long)
        }
    

class LitDataLoader(pl.LightningDataModule):
    
    def __init__(self, dataset: "td.Dataset", batch_size: Optional[int]=64):

        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self):
        pass
    
    def train_dataloader(self) -> "td.DataLoader":
        return td.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)
    
    def val_dataloader(self) -> "td.DataLoader":
        return td.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)
    
    def test_dataloader(self) -> "td.DataLoader":
        return td.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)
    
    













