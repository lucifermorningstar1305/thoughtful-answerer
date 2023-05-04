import numpy as np
import pandas as pd
import transformers
import torch
import torch.utils.data as td
import faiss
import os

import argparse

from rich import print as rprint
from rich.progress import track

if __name__ == "__main__":

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()

    parser.add_argument("--evidence_path", "-e", required=True, type=str, help="the path where your evidences are stored.")
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
    rprint(f"Total number of evidences: {len(evidences)}")


    ## Load the context encoder of the DPR model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ctx_tokenizer = transformers.DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", 
                                                                                do_lower_case=True)
    
    ctx_encoder = transformers.DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base",
                                                                 model_type="DPRContextEncoder").to(device)
    

    ## Tokenize the Evidences and build a dataloader of it.
    ctx_input_ids = ctx_tokenizer(evidences, padding="max_length", truncation=True,
                                  max_length=args.max_length)["input_ids"]
    
    assert len(ctx_input_ids) == len(evidences), "Error :( the Context Encoder tokenized input lengths does not match the length of the provided Evidences."

    ctx_tensor = torch.tensor(ctx_input_ids, dtype=torch.int)
    ctx_dataset = td.TensorDataset(ctx_tensor)
    ctx_dataloader = td.DataLoader(ctx_dataset, batch_size=args.batch_size, 
                                   shuffle=False, pin_memory=True, num_workers=4)
    

    ## Create Faiss index
    # nlist = 5
    # faiss_quantizer = faiss.IndexFlatL2(768) 
    # faiss_index = faiss.IndexIVFFlat(faiss_quantizer, 768, nlist)
    faiss_index = faiss.IndexFlatIP(768)

    ## Training the Faiss Index
    if not faiss_index.is_trained:
        rprint("[bold #00A550] Training Faiss Index")
        for batch in track(ctx_dataloader):
            x = batch[0].to(device)

            with torch.no_grad():
                embedding = ctx_encoder(x).pooler_output
                faiss_index.train(embedding.detach().cpu().numpy())


    ## Store the embeddings into the Faiss Index
    if faiss_index.is_trained:
        rprint("[bold #3AA8C1] Building Context Vectors")
        for batch in track(ctx_dataloader):
            x = batch[0].to(device)

            with torch.no_grad():
                embedding = ctx_encoder(x).pooler_output
                faiss_index.add(embedding.detach().cpu().numpy().astype(np.float32))

        assert len(evidences) == faiss_index.ntotal, f"Sorry the Number of evidences : {len(evidences)} != number of totals in faiss index : {faiss_index.ntotal}"

        ## Save the Faiss index to the specified location
        save_loc = os.path.join(args.save_dir, args.file_name)
        rprint(f"Saving the faiss index to : {save_loc}")
        faiss.write_index(faiss_index, save_loc)



    