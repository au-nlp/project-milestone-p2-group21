from datasets import load_dataset
import googletrans
from googletrans import Translator
import asyncio

en_yue_sentences = load_dataset("google/smol", "smolsent__en_yue")


languages = ["chinese (simplified)", "chinese (traditional)", "cantonese"]
destination = "english"

async def translate_text(text, src_lang, dest_lang):
    async with Translator() as translator:
        result = await translator.translate(text, src = googletrans.LANGCODES[src_lang], dest = googletrans.LANGCODES[dest_lang])
        return result

all_translations = []

import nltk
from tqdm import tqdm

sent = en_yue_sentences["train"]["trg"][0]
lan = languages[0]

for sentence in tqdm(en_yue_sentences["train"].select(range(100))):
    translations = {language: asyncio.run(translate_text(sentence["trg"], language, destination)).text for language in languages }
    translations["original"] = sentence["src"]
    translations["original_yue"] = sentence["trg"]
    for language in languages:
        translations[language + "_bleu"] = nltk.translate.bleu_score.sentence_bleu([translations["original"].split(" ")], translations[language].split(" "))
    all_translations.append(translations)


import numpy as np
def print_all_bleu(all_translations):
    print(np.mean([translation["chinese (simplified)_bleu"] for translation in all_translations]))
    print(np.mean([translation["chinese (traditional)_bleu"] for translation in all_translations]))
    print(np.mean([translation["cantonese_bleu"] for translation in all_translations]))


def print_trans(trans):
    print(f"original_yue          :         {trans['original_yue']}")
    print(f"original              :         {trans['original']}")
    print(f'chinese_simplified    : ({trans["chinese (simplified)_bleu"]:.2f}): {trans["chinese (simplified)"]}')
    print(f'chinese (traditional) : ({trans["chinese (traditional)_bleu"]:.2f}): {trans["chinese (traditional)"]}')
    print(f'cantonese             : ({trans["cantonese_bleu"]:.2f}): {trans["cantonese"]}')
