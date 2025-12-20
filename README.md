[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

# Improving Cantonese translation in existing multilingual translation models
## Abstract
Cantonese is a low-resource language, i.e. it has received a very low amount of attention in NLP research compared to the number of speakers. Recently, high-quality datasets such as SMOL and Gatitos have been released, providing professionally translated parallel data on multiple levels (token-level, sentence-level, and document-level) for low-resource languages like Cantonese. This project explores how to utilize these datasets to improve the performance of a pre-trained multilingual translation model, MADLAD, on Cantonese to English translation tasks. For the token-level data we try to exploit the similarity to written Mandarin, on which models perform much better, by linearly transforming Cantonese embeddings to MADLAD's Mandarin embedding space, but this made no improvement in practice. For the sentence- and document-level data we show that using it to finetune MADLAD through backtranslation significantly improves model performance (6.46% improvement in BLEURT score).


## Contributions
This project
- Demonstrates that Cantonese-only tokens degrade the performance of models on Cantonese-translation tasks, and explores how token- and embedding-transformation can improve performance in these cases.
- Gives a method for utilizing Gatitos' character level translations, SMOLSent sentence level translations and SMOLDoc document level translations to train Cantonese -> English translation models.
- Shows that using the SMOL dataset to finetune Cantonese -> English models improves translation performance.

## Additional Datasets
In order to train word2vec embeddings for Cantonese we use unlabelled Cantonese data. For this, we use the monolingual Cantonese corpus compiled by M. Dare, et. al. that sources data from other monolingual datasets, Wikipedia, Youtube Subtitles, Instagram and Cantonese blogs. The dataset consists of 923084 unlabelled sentences with spaces inserted between tokens. Without inserted spaces, the average length of sentences in the dataset is 33 characters, the median length is 25 characters and the max length is 1433 characters. The dataset appears to be much higher quality than other publicly available monolingual datasets, such as the internet-scraped MADLAD which contains lots of english text, website names, dates, urls and general website information. Two randomly selected sentences from the dataset can be seen below: 

      有人 用 毛 代 角 。

      第二 階段 投票 將會 喺 下 周 一 至 周 五 舉行 ， 從 最後 五 強中 票選 「 我 最喜愛 的 男 / 女 / 組合 」 ， 結果 將會 喺 1 月 1 號 嘅 頒獎禮 公佈 。

Also, we use the Gatitos dataset, also from Google. This dataset contains token level for translations 179 pairs of languages, including 4.2k rows of Cantonese to Mandarin character translations.


## Methods
For initial benchmarking of the state of the art we scrape SMOLSent translations from Google & Bing Cantonese and Simplified/Traditional Chinese models. For initial evaluation of translation quality we use BLEURT as an evaluation proxy of human assessment. For in depth analysis we do manual evaluation of the English-outcome translations.

Since SMOLSent and SMOLDoc consists of English -> Cantonese translations we utilize backtranslation, translating from the target to the source, in order to train Cantonese -> English models. 

## Timeline
07/11/2025 - 14/11/2025 - Collecting unlabelled Cantonese dataset.

14/11/2025 - 21/11/2025 - Train word2vec embeddings on unlabelled cantonese data. 

21/11/2025 - 28/11/2025 - Perform embedding transform on trained word2vec embeddings using Gatitos character dataset 

28/11/2025 - 05/12/2025 - Do back-translation finetuning on SMOLSent English -> Cantonese and SMOLDoc English -> Cantonese. 

05/12/2025 - 19/12/2025 - Write and submit final project and report. 
## Organization (Milestones)
1. Collect corpus of unlabelled Cantonese data. 
2. Train word2vec embeddings on unlabelled Cantonese data.
3. Perform embedding transform on Cantonese word2vec embeddings to our model's (Madlad?) embeddings, using Gatitos character-level translations.
4. Do back-translation finetuning on SMOLSent English -> Cantonese and SMOLDoc English -> Cantonese.
5. Potentially expand approach to other low resource languages.

## Codebase
An overview of the experiments can be found in the file `main.ipybn`

## Individual Contributions
### Victor
### Lasse
Found GPU compute resources, finetuned MADLAD model, did inference on different models (original, finetuned and with transformation matrix) to obtain baselines on original vs tokenswapped data, web scraping.
### Andreas
Theory and model exploration, experiment evaluation, data processing, and report writing.
