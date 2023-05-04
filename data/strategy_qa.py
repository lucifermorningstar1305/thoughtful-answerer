from typing import Any, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import json
import random

from rich.progress import track
from rich import print as rprint

from collections import defaultdict

def parse_evidence(evidences: List, para_data: Dict) -> List:
    """
    This function parses the evidence of the StrategyQA dataset.

    :params evidence: the list of evidence for the question-answers.

    :returns: the parsed evidences.
    """

    parsed_evidences = list()

    assert type(evidences) == list, "Obtained evidences is not a list"
    assert type(para_data) == dict, "Obtained paragraphs is not a dictionary"

    for evidence in evidences:
        for annotations in evidence:
            for annot in annotations:
               if type(annot) == list:
                   for key in annot:
                       parsed_evidences.append(para_data[key]["content"])


    return parsed_evidences
                       
                   




if __name__ == "__main__":

    with open("./strategyqa_dataset/strategyqa_train.json", "r") as fp:
        train_data = json.load(fp)

    with open("./strategyqa_dataset/strategyqa_train_paragraphs.json", "r") as fp:
        train_para_data = json.load(fp)

    final_data = defaultdict(lambda: list())
    evidences = list()

    for  data in track(train_data):
        final_data["question"].append(data["question"])
        final_data["facts"].append(data["facts"])
        final_data["answer"].append(data["answer"])
        evidences.extend(parse_evidence(data["evidence"], train_para_data))


    final_data = pd.DataFrame(final_data)
    final_data.to_csv("./strategyqa_dataset/train.csv", index=False)
    random.shuffle(evidences)
    evidences = "\n".join(evidences)
    
    with open("./strategyqa_dataset/evidences.txt", "w") as fp:
        fp.write(evidences)





    
