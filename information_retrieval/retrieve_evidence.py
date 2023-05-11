import numpy as np
import pandas as pd
import faiss
import transformers
import torch
import os
import sys
import argparse
import json

from utils import EmbeddingGenerator, FaissIDX

from rich import print as rprint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", "-i", required=True, type=str, help="the hugging face model id")
    parser.add_argument("--faiss_loc", "-f", required=True, type=str, help="location of the faiss indices.")
    parser.add_argument("--evidence_loc", "-e", required=True, type=str, help="location of the evidences.")
    parser.add_argument("--top_k", "-t", required=False, type=int, default=10, help="number of evidences to return for a given query.")
    
    args = parser.parse_args()

    ques = "Are more people related to Genghis Khan than Julius Caesar?"

    ## Load the evidences
    rprint("[bold #FF0054]Loading evidences ...")
    with open(args.evidence_loc, "r") as fp:
        evidences = fp.read()
        
    evidences = evidences.split("\n")

    embedding_model = EmbeddingGenerator(args.model_id)
    faiss_indexer = FaissIDX(embedding_model, load_chkpt=args.faiss_loc)

    rprint(faiss_indexer.search_doc(ques, evidences, k=args.top_k))
