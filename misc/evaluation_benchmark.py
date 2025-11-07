import json

from bleurt import score
import numpy as np
from tqdm import tqdm

with open("sentence_translations_bing.json") as f:
    trans_bing = json.load(f)

with open("sentence_translations_google.json") as f:
    trans_google = json.load(f)

scorer = score.BleurtScorer("bleurt-base-128")

#def print_all_bleu(trans_bing, trans_google):

google_simplified_chinese_scores = [scorer.score(references=[trans["original"]], candidates=[trans["chinese"]]) for trans in tqdm(trans_google, desc="evaluating simplified chinese (google)")]
bing_cantonese_scores = [scorer.score(references=[trans["original"]], candidates=[trans["cantonese"]]) for trans in tqdm(trans_bing, desc="evaluating cantonese (bing)")]
bing_simplified_chinese_scores = [scorer.score(references=[trans["original"]], candidates=[trans["chinese (simplified)"]]) for trans in tqdm(trans_bing, desc="evaluating simplified chinese (bing)")]
bing_traditional_chinese_scores = [scorer.score(references=[trans["original"]], candidates=[trans["chinese (traditional)"]]) for trans in tqdm(trans_bing, desc="evaluating traditional chinese (bing)")]


print(np.mean(google_simplified_chinese_scores))
print(np.mean(bing_cantonese_scores))
print(np.mean(bing_simplified_chinese_scores))
print(np.mean(bing_traditional_chinese_scores))



all_trans = []

for x in trans_bing:
    for z in trans_google:
        if z["original"] == x["original"]:
            all_trans.append(
                {
                    "original": z["original"],
                    "bing": x,
                    "google": z
                }
            )