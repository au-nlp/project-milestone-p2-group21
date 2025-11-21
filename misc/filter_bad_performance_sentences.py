import json
import numpy as np

with open("all_trans_bleurt.json", "r") as f:
    all_trans = json.load(f)

lang_bing = ["cantonese", "chinese (simplified)", "chinese (traditional)"]
lang_google = ["chinese"]

bad_sentences = []

for trans in all_trans[:10]:
    bleu_scores = [trans["bing"][lang + "_bleu"] for lang in lang_bing] + [trans["google"][lang + "_bleu"] for lang in lang_google]
    bleurt_scores = [trans["bing"][lang + "_bleurt"] for lang in lang_bing] + [trans["google"][lang + "_bleurt"] for lang in lang_google]
    if np.mean(bleu_scores) < 0.3 and np.mean(bleurt_scores) > 0.5:
    #if np.all(np.array(scores) < -0.3 ):
        bad_sentences.append(trans)

def print_sentences(sent_trans):
    print(f' original : {sent_trans["original"]}')
    [print(f' bing {lang} : {sent_trans["bing"][lang]}') for lang in lang_bing]
    [print(f' google {lang} : {sent_trans["google"][lang]}') for lang in lang_google]
