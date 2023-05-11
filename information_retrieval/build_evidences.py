from typing import List

import numpy as np
import matplotlib.pyplot as plt
import transformers
import torch
import torch.utils.data as td
import faiss
import os
import re
import unicodedata

import argparse

from rich import print as rprint
from rich.progress import track


from utils import EmbeddingGenerator, FaissIDX

def preprocess(doc: str):

    doc = unicodedata.normalize('NFKD', doc)
    doc = doc.lower()
    doc = doc.strip("\n")
    doc = re.sub(r"[^\w\s\d]", "", doc)

    return doc



if __name__ == "__main__":

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", "-i", required=True, type=str, help="the hugging face model id")
    parser.add_argument("--evidence_path", "-e", required=True, type=str, help="the path where your evidences are stored.")
    parser.add_argument("--embedding_size", "-E", required=True, type=int, help="the embedding size returned by the model.")
    parser.add_argument("--save_dir", "-s", required=False, type=str, default="./", help="path to save file.")
    parser.add_argument("--file_name", "-f", required=False, type=str, default="faiss_index.npy", help="filename for the saving the file.")
    parser.add_argument("--max_length", "-m", required=False, type=int, default=512, help="maximum length of the tokens")
    parser.add_argument("--batch_size", "-b", required=False, type=int, default=16, help="the batch size for processing the evidences.")

    args = parser.parse_args()


    if args.save_dir != "./" and not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)


    ## Load the evidences
    with open(args.evidence_path, "r") as fp:
        evidences = fp.read()

    evidences = evidences.split("\n")
    evidences = list(map(lambda x: preprocess(x), evidences))

    rprint(f"Total number of evidences: {len(evidences)}")

    embedding_model = EmbeddingGenerator(args.model_id)
    faiss_indexer = FaissIDX(embedding_model, dim=args.embedding_size, 
                            save_chkpt=os.path.join(args.save_dir, args.file_name))

    for doc in track(evidences):
        faiss_indexer.add_doc(doc)

    faiss_indexer.save_index()



    