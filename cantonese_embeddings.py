from pathlib import Path
import os

from gensim.models import Word2Vec
import numpy as np
from nltk.corpus import brown, stopwords

with open(Path("./") / "unlabelled" / "cantonese.txt", "r", encoding="utf8") as f:
    text = f.read()

sentences = [x.split(" ") for x in text.split("\n")]
#stopwords = set(stopwords.words("chinese"))

def train_word2vec(sentences):
    model = Word2Vec(
        sentences=sentences,
        vector_size=1024,   # embedding size 
        window=3,         # context window
        min_count=1,      # keep all words 
        workers=4,        # number of CPU cores to use
        sg=1,             # skip-gram (better for small data)
        epochs=1,        # more passes to learn something
        seed=42
    )

    return model

if os.path.exists((Path("./") / "unlabelled" / "1_epoch_embeddings").__str__()):
    model = Word2Vec.load((Path("./") / "unlabelled" / "1_epoch_embeddings").__str__())
else: 
    model = train_word2vec(sentences)
#model.save((Path("./") / "unlabelled" / "cantonese_embeddings.model").__str__())
model.save((Path("./") / "unlabelled" / "1_epoch_embeddings").__str__())

# Embedding arithmetic tests
def find_best(similar_words, input_words):
    similar_words = [(x,y) for x,y in similar_words if x not in input_words]
    max_index = np.argmax([y for x,y in similar_words])
    return similar_words[max_index][0]

# 爸爸 (dad) - 男 (man) + 女 (woman) = 媽媽 (mom) 
#print(model.wv.similar_by_vector(model.wv["爸爸"] - model.wv["男"]*2 + model.wv["女"]*2))
print(f'爸爸 - 男 + 女 = {find_best(model.wv.similar_by_vector(model.wv["爸爸"] - model.wv["男"] +  model.wv["女"]), ["爸爸", "男", "女"])}')

# 王 (king) - 男 (man) + 女 (woman) = 女皇 / 皇帝 (queen) 
#print(model.wv.similar_by_vector(model.wv["王"] - model.wv["男"] +  model.wv["女"]))
print(f'王 - 男 + 女 = {find_best(model.wv.similar_by_vector(model.wv["王"] - model.wv["男"] +  model.wv["女"]), ["王", "男", "女"])}')


# 香港 (hong kong) - 城市 (city) + 國家 (country) = 中國 / 中國大陸 (china)
#print(model.wv.similar_by_vector(model.wv["香港"] - model.wv["城市"] +  model.wv["國家"]))
print(f'香港 - 城市 + 國家 = {find_best(model.wv.similar_by_vector(model.wv["香港"] - model.wv["城市"] +  model.wv["國家"]), ["香港", "城市", "國家"])}')


# 電動火車 (eletric train) - 電 (electricity) + 煤炭 (coal) = 蒸汽火車 (steam locomotive)  
combined_word_steam_locomotive = model.wv.n_similarity([model.wv["蒸汽"] + model.wv["火車"]], [model.wv["電動"]  + model.wv["火車"]- model.wv["電"] + model.wv["煤炭"]])
print(f'電動火車 - 電 + 煤炭 = {find_best(model.wv.similar_by_vector(model.wv["電動"] + model.wv["火車"] - model.wv["電"] +  model.wv["煤炭"]) + [("蒸汽火車", combined_word_steam_locomotive)], ["電動火車", "電", "煤炭"])}')




