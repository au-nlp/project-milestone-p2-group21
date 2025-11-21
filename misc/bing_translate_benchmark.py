from datasets import load_dataset
import translators as ts
import time

en_yue_sentences = load_dataset("google/smol", "smolsent__en_yue")

languages = ["chinese (simplified)", "chinese (traditional)", "cantonese"]
lang_map = {"chinese (simplified)" : "zh-Hans", "chinese (traditional)": "zh-Hant", "cantonese": "yue", "english": "en"}
service_urls = ['translate.google.com']
destination = "english"

all_trans = []

#test
#ts.translate_text(en_yue_sentences["train"][0]["trg"], translator="bing", from_language=lang_map["cantonese"], to_language=lang_map["english"])


import nltk
from tqdm import tqdm

for sentence in tqdm(en_yue_sentences["train"]):
    translations = {}
    attempts = 0
    while attempts < 3:
        try:
            translations = {language: ts.translate_text(sentence["trg"], translator="bing", from_language=lang_map[language], to_language=lang_map[destination]) for language in languages }
            break
        except:
            tqdm.write(f"error on sentece {sentence['id']}, retrying... ({attempts + 1})")
            attempts += 1
            time.sleep(3)
    if attempts == 3:
        tqdm.write(f"skipped sentence {sentence['id']} after {attempts+1} attempts")
        continue

    translations["original"] = sentence["src"]
    translations["original_yue"] = sentence["trg"]
    for language in languages:
        translations[language + "_bleu"] = nltk.translate.bleu_score.sentence_bleu(
            [translations["original"].split(" ")], 
            translations[language].split(" "), 
                #use smoothing method 7 that had best chinese->english human-evaluation corerlation from https://aclanthology.org/W14-3346/
            smoothing_function= nltk.translate.bleu_score.SmoothingFunction().method7 
            )
        
    all_trans.append(translations)


import numpy as np
def print_all_bleu(translations):
    [print(f'{lang}: {np.mean([[trans[lang + "_bleu"] for trans in translations]])}') for lang in languages]


def print_trans(trans):
    print(f"original_yue          :         {trans['original_yue']}")
    print(f"original              :         {trans['original']}")
    print(f'chinese_simplified    : ({trans["chinese (simplified)_bleu"]:.2f}): {trans["chinese (simplified)"]}')
    print(f'chinese (traditional) : ({trans["chinese (traditional)_bleu"]:.2f}): {trans["chinese (traditional)"]}')
    print(f'cantonese             : ({trans["cantonese_bleu"]:.2f}): {trans["cantonese"]}')
