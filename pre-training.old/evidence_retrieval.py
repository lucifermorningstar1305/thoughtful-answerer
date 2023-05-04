import numpy as np
import pandas as pd

from rich import print as rprint
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes import DensePassageRetriever

if __name__ == "__main__":
    
    doc_store = ElasticsearchDocumentStore(
        host="localhost",
        username="",
        password="",
        index="strategyqa"
    )

    retriever = DensePassageRetriever(
        document_store=doc_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu = True,
        embed_title = True
    )

    rprint(retriever.retrieve('Would a dog respond to bell before Grey seal?'))