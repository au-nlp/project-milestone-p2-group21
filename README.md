[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

# ?Cantonese? ?ML? ?Title?
## Abstract
Most NLP research focusses on English and other "high resource" languages. This means that "low ressource" often show much worse results in all NLP tasks. The SMOL dataset aims to lessen this difference by providing high quality sentence- and document-level translations for many low ressource languages. This projects uses SMOL and Gatitos and focuses on Cantonese, which is the second most spoken Sinitic language. Despite having more than 85 million speakers it is severely underrepresented in the NLP field. We attempt to analyse shortcommings of existing models and finetune these to relieve the shortcommings. Our initial analasis shows that doing a simple token replacement of Cantonese characters with GAITITOS and then translating with existing models yields better results. Based on this, we believe that various embedding-transformation approaches will result in significant Cantonese-translation improvement for existing models.  
## Contributions
Our contributions are:
* A way to efficiently utilize Gatitos' character level translations, SMOLSent sentence level translations and SMOLDoc document level translations to train Cantonese -> English translation models.
* Demonstrating that Cantonese-only tokens degrade the performance of models on Cantonese-translation tasks, and token- and embedding-transformation approaches that significantly improve performance in these cases.
* Showing that using the SMOL dataset to finetune Cantonese -> English models improves translation performance. 
## Potential extra datasets
For future finetuning, basede on the findings of Dare et. al., there's large potential in giving models more data, even unlabled, as models have usually been exposed to very little Cantonese data. They use a corpus consisting of Wikipedia articles and a cleaned scrape of Instagram and Youtube subtitles, and have compiled this into a dataset that might be useful in our project too.

In order to train word2vec embeddings for Cantonese we collect unlabelled Cantonese data. We use the monolingual Cantonese corpus compiled by M. Dare, et. al that sources data from other monolingual datasets, Wikipedia, Youtube Subtitles, Instagram and Cantonese blogs. 
## Methods
**TODO: Write full text not bullet points**
- Benchmarking of existing models and analsis of areas where these show weak performance
- Finetuning models with additional data from SMOL
## Timeline
7/11/2025 - 14/11/2025 - Collecting unlabelled Cantonese dataset.

14/1172025 - 21/11/2025 - Train word2vec embeddings on unlabelled cantonese data. 

21/11/2025 - 28/11/2025 - Perform embedding transform on trained word2vec embeddings using Gatitos character dataset 

28/11/2025 - 5/12/2025 - Do back-translation finetuning on SMOLSent English -> Cantonese and SMOLDoc English -> Cantonese. 

5/12/2025 - 19/12/2025 - Write and submit final project and report. 
## Organization
1. Collect corpus of unlabelled Cantonese data. 
2. Train word2vec embeddings on unlabelled Cantonese data.
3. Perform embedding transform on Cantonese word2vec embeddings to our model's (Madlad?) embeddings, using Gatitos character-level translations.
4. Do back-translation finetuning on SMOLSent English -> Cantonese and SMOLDoc English -> Cantonese.
5. Potentially expand approach to other low resource languages.

## References
      Unsupervised Mandarin-Cantonese Machine Translation
      Megan Dare and Valentina Fajardo Diaz and Averie Ho Zoen So and Yifan Wang and Shibingfeng Zhang
      https://arxiv.org/abs/2301.03971  

      MADLAD-400: A Multilingual And Document-Level Large Audited Dataset
      Sneha Kudugunta and Isaac Caswell and Biao Zhang and Xavier Garcia and Christopher A. Choquette-Choo and Katherine Lee and Derrick Xin and Aditya Kusupati and Romi Stella and Ankur Bapna and Orhan Firat  
      https://arxiv.org/abs/2309.04662
## Appendix
### Repo org
main.ipybn contains data exploration and benchmarks of existing models, both traditional and a T5 based model.
### Questions
