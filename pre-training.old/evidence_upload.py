# This script is used to upload the evidences of the StrategyQA dataset to
# an elasticsearch cluster running in the localhost.

import numpy as np
import pandas as pd
import transformers
import os
import sys
import requests

from rich.progress import track
from rich import print as rprint

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore

if __name__ == "__main__":

    with open("./data/strategyqa_dataset/evidences.txt", "r") as fp:
        evidences = fp.read()

    evidences = evidences.split("\n")

    # rprint(requests.get("http://localhost:9200/_cluster/health").json())
    # rprint(requests.get('http://localhost:9200/_cat/indices').text)

    doc_store = ElasticsearchDocumentStore(
        host="localhost",
        username="",
        password="",
        index="strategyqa"
    )

    evidences_json = [
        {
            'content' : para,
            'meta' : {
                'source' : 'Wikipedia'
            }
        } for para in evidences
    ]

    rprint(evidences_json[:5])

    doc_store.write_documents(evidences_json)

    

