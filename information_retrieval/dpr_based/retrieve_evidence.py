import numpy as np
import pandas as pd
import faiss
import transformers
import torch
import os
import sys
import argparse
import json

from rich import print as rprint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--faiss_loc", "-f", required=True, type=str, help="location of the faiss indices.")
    parser.add_argument("--evidence_loc", "-e", required=True, type=str, help="location of the evidences.")
    parser.add_argument("--top_k", "-t", required=False, type=int, default=10, help="number of evidences to return for a given query.")
    
    args = parser.parse_args()

    ques = "Would a dog respond to bell before Grey seal?"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rprint(f"Device in use: {device}")


    # Load the evidences
    rprint("[bold #FF0054]Loading evidences ...")
    with open(args.evidence_loc, "r") as fp:
        evidences = fp.read()

    evidences = evidences.split("\n")

    # Setup the encoder for the question
    rprint("[bold #9E0059]Generating Question embedding ...")
    ques_tokenizer = transformers.DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-single-nq-base",
                                                                                  do_lower_case=True)
    ques_encoder = transformers.DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)

    # Tokenize the ques and get the embedding
    ques_tokens = ques_tokenizer(ques, max_length=512, padding="max_length", truncation=True, 
                                 add_special_tokens=True, return_tensors="pt")["input_ids"]
    
    ques_embedding = ques_encoder(ques_tokens.to(device))["pooler_output"].detach().cpu().numpy()


    rprint(f"Shape of the ques embedding : {ques_embedding.shape}")


    # Load Faiss Index
    rprint("[bold #FFBD00]Loading Faiss Indices ...")
    faiss_index = faiss.read_index(args.faiss_loc)
    
    # Get indices of the nearest
    rprint("[bold #390099]Searching for the top_k evidences ...")
    top_k = args.top_k
    faiss.normalize_L2(ques_embedding)
    D, I = faiss_index.search(ques_embedding, top_k)
    D = D.reshape(-1)
    I = I.reshape(-1)

    # List Evidences
    results = {}
    for i, (score, idx) in enumerate(zip(D, I)):
        results[f"evidence-{i+1}"] = {'score': score, 'evidence':evidences[idx]}

    rprint(results)


