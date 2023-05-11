from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import faiss
import transformers
import os
import sys


class EmbeddingGenerator:
    def __init__(self, model_name: str, max_length: Optional[int]=512, truncation: Optional[bool] = True):
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.truncation = truncation
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)


    def get_embeddings(self, ctx: str):

        inputs = self.tokenizer(ctx, padding="max_length", max_length=self.max_length, 
                                    truncation=self.truncation, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            embedding = self.model(**inputs)['pooler_output']

        return embedding.detach().cpu().numpy()
    

class FaissIDX:

    def __init__(self, model: "EmbeddingGenerator", dim: Optional[int]=768, save_chkpt: Optional[str]=None, 
                 load_chkpt: Optional[str]=None):
        
        self.model = model
        self.save_chkpt = save_chkpt
        
        if load_chkpt is None:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.read_index(load_chkpt)

    def add_doc(self, doc: str):
        embedding = self.model.get_embeddings(doc)

        faiss.normalize_L2(embedding)
        self.index.add(embedding)


    def search_doc(self, query:str, docs: List, k: Optional[int]=10):
        
        query_embedding = self.model.get_embeddings(query)
        faiss.normalize_L2(query_embedding)

        D, I = self.index.search(query_embedding, k=k)
        return [{docs[idx]: score} for idx, score in zip(I[0], D[0])]
    
    def save_index(self):
        faiss.write_index(self.index, self.save_chkpt)



    


    


        

