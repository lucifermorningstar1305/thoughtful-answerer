import pandas as pd
import openai
import backoff
import requests
import os
import sys

import time
import argparse
import configparser

from rich import print as rprint
from rich.progress import track


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completion(config_path, **kwargs):

    config = configparser.ConfigParser()
    config.read(config_path)

    openai.api_key = config["KEYS"]["OPENAPI"]

    resp = openai.Completion.create(**kwargs)
    return resp["choices"][0]["text"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", "-f", required=True, type=str, help="The file path of the csv/json data")
    parser.add_argument("--prompt_path", "-p", required=True, type=str, help="The path of the prompts")
    parser.add_argument("--config_path", "-c", required=True, type=str, help="The path of the config file")
    parser.add_argument("--save_path","-s", type=str, help="The path to save the file along with the filename")

    args = parser.parse_args()
    
    file_type = args.file_path.split(".")[-1]

    if file_type != "csv":
        raise Exception(f"Expected a csv file. Found a {file_type} file.")
    
    df = pd.read_csv(args.file_path)

    with open(args.prompt_path, "r") as fp:
        context = fp.read()

    df['cot_answer'] = None
    cots = list()

    for ques in track(df["question"].values):
        
        ques = ques.strip("\n")
        ques = f"\nQ:{ques}"
        resp = completion(args.config_path, 
                          engine="text-davinci-002", prompt=context+ques,
                          max_tokens=1024, temperature=.7, top_p=1, frequency_penalty=0.2, presence_penalty=0.25)
        
        cots.append(resp.strip("\n"))

    df["cot_answers"] = cots

    if args.save_path is not None:
        df.to_csv(args.save_path, index=False)

    else:
        df.to_csv(args.file_path, index=False)

        

        



