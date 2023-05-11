# from typing import List, Tuple, Any, Optional, Dict
# from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

# import numpy as np
# import pandas as pd
# import transformers
# import torch
# import torch.utils.data as td
# import faiss
# import os

# from rich.progress import track
# from rich import print as rprint


# if __name__ == "__main__":

#     torch.cuda.empty_cache()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     rprint(f"Using device : {device}")

#     with open("../data/strategyqa_dataset/evidences.txt", "r") as fp:
#         evidences = fp.read()

#     ctx_tokenizer = transformers.DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base",
#                                                                                 do_lower_case=True)
    
#     ctx_encoder = transformers.DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base",
#                                                                  model_type="DPRContextEncoder").to(torch.device("cuda"))

    
    
#     evidences = evidences.split("\n")
#     embeddings = list()

#     input_ids = ctx_tokenizer(evidences, padding="max_length", truncation=True,
#                                                 max_length=512)["input_ids"]
    
#     assert len(input_ids) == len(evidences), "The length of the evidences does not match the length of the input_ids"

#     input_ids = torch.tensor(input_ids, dtype=torch.int)
#     ctx_dataset = td.TensorDataset(input_ids)
#     ctx_dataloader = td.DataLoader(ctx_dataset, batch_size=8, num_workers=4, pin_memory=True, shuffle=False)

#     index = faiss.IndexFlatL2(768)

#     for batch in track(ctx_dataloader):
#         x = batch[0].to(device)
#         with torch.no_grad():
#             embed = ctx_encoder(x).pooler_output
#             index.add(embed.detach().cpu().numpy())

#     rprint(f"Total number of context in faiss index : {index.ntotal}")
#     rprint(f"Writing faiss index to ./faiss_index.npy")
#     faiss.write_index(index, "./faiss_index.npy")

        

import requests

API_URL = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
# API_URL = "https://api-inference.huggingface.co/models/nomic-ai/gpt4all-j"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "<|prompter|>What is a meme, and what's the history behind this word?<|endoftext|><|assistant|>",
	"parameters" : {
		"temperature" : 1.,
		"repetition_penalty" : 1.,
		"return_full_text" : False,
		"do_sample" : True,
		"max_new_tokens" : 250
    },
    "options" : {
	    "use_cache" : True
    }
})

print(output)
    


