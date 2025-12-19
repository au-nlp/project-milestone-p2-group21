from pathlib import Path
import os

from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
#from nltk.corpus import brown, stopwords

"""

with open(Path("./") / "unlabelled" / "cantonese.txt", "r", encoding="utf8") as f:
    text = f.read()

sentences = text.split("\n")

from transformers import T5Tokenizer

model_name = 'jbochi/madlad400-3b-mt'
#model = T5ForConditionalGeneration.from_pretrained(model_name, device_map=None)
tokenizer = T5Tokenizer.from_pretrained(model_name)


sentences = ["".join(x.split(" ")) for x in tqdm(sentences)]
sentences = [tokenizer(x) for x in tqdm(sentences)]
sentences = [[tokenizer.decode(z) for z in x["input_ids"]] for x in tqdm(sentences)]

#sentences = [x.split(" ") for x in tqdm(sentences[:int(len(sentences)*0.1)])]
#sentences = [x for x in sentences if len(x) >= 10]
#stopwords = set(stopwords.words("chinese"))
"""

from gensim.models.callbacks import CallbackAny2Vec


losses = []

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.last_loss = 0

    def on_epoch_end(self, model):
        global losses
        loss = model.get_latest_training_loss()
        loss = loss - self.last_loss
        self.last_loss += loss
        losses.append(loss)
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

def train_word2vec(sentences):
    model = Word2Vec(
        sentences=sentences,
        vector_size=1024,   # embedding size 
        window=10,         # context window
        min_count=1,      # keep all words 
        workers=1,        # number of CPU cores to use
        sg=1,             # skip-gram (better for small data)
        epochs=6,        # more passes to learn something
        seed=42,
        compute_loss=True,
        #alpha=0.1,
        #min_alpha=0.000001,
        #shrink_windows=True,
        callbacks=[callback()],
        #negative=5,
        #ns_exponent=0.75,
        #sample=10**-5,
        #hs=0
    )

    return model

#model = train_word2vec(sentences)
#model.save((Path("./") / "unlabelled" / "character_embeddings_6epoch.model").__str__())
model = Word2Vec.load((Path("./") / "unlabelled" / "character_embeddings_6epoch.model").__str__())

import matplotlib.pyplot as plt

plt.plot(losses)
plt.show()

# Embedding arithmetic tests
def find_best(similar_words, input_words):
    similar_words = [(x,y) for x,y in similar_words if x not in input_words]
    max_index = np.argmax([y for x,y in similar_words])
    return similar_words[max_index][0]

def average_embedding(chars):
    return np.mean([model.wv[char] for char in chars],axis=0)

# 爸爸 (dad) - 男 (man) + 女 (woman) = 媽媽 (mom) 
#print(model.wv.similar_by_vector(model.wv["爸爸"] - model.wv["男"]*2 + model.wv["女"]*2))
print(f'爸爸 - 男 + 女 = {model.wv.similar_by_vector(average_embedding("爸爸") - average_embedding("男") +  average_embedding("女"))}')

print(f'爸爸 - 男 + 女 = {find_best(model.wv.similar_by_vector(model.wv["爸爸"] - model.wv["男"] +  model.wv["女"]), ["爸爸", "男", "女"])}')

# 王 (king) - 男 (man) + 女 (woman) = 女皇 / 皇帝 (queen) 
#print(model.wv.similar_by_vector(model.wv["王"] - model.wv["男"] +  model.wv["女"]))
print(f'王 - 男 + 女 = {find_best(model.wv.similar_by_vector(model.wv["王"] - model.wv["男"] +  model.wv["女"]), ["王", "男", "女"])}')
#model.wv.similar_by_vector(average_embedding("王") - average_embedding("男") +  average_embedding("女"))


# 香港 (hong kong) - 城市 (city) + 國家 (country) = 中國 / 中國大陸 (china)
#print(model.wv.similar_by_vector(model.wv["香港"] - model.wv["城市"] +  model.wv["國家"]))
print(f'香港 - 城市 + 國家 = {find_best(model.wv.similar_by_vector(model.wv["香港"] - model.wv["城市"] +  model.wv["國家"]), ["香港", "城市", "國家"])}')

#model.wv.similar_by_vector(average_embedding("香港") - average_embedding("城市") +  average_embedding("國家"))


# 電動火車 (eletric train) - 電 (electricity) + 煤炭 (coal) = 蒸汽火車 (steam locomotive)  
combined_word_steam_locomotive = model.wv.n_similarity([model.wv["蒸汽"] + model.wv["火車"]], [model.wv["電動"]  + model.wv["火車"]- model.wv["電"] + model.wv["煤炭"]])
print(f'電動火車 - 電 + 煤炭 = {find_best(model.wv.similar_by_vector(model.wv["電動"] + model.wv["火車"] - model.wv["電"] +  model.wv["煤炭"]) + [("蒸汽火車", combined_word_steam_locomotive)], ["電動火車", "電", "煤炭"])}')




