#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:10:11 2024
@author: jason

This code is to generate a word cloud from a list of research topics.


"""
import os
import pandas as pd
import numpy as np
import json
import copy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# step 1: get topic list from excel and clean the data
excel_name = "D:/research topic word cloud/research ppl profile APSS_jc.xlsx"
df = pd.read_excel(excel_name, header=None)
topic_list_raw = df.loc[:, 2].tolist()
topic_list_raw = [i for i in topic_list_raw if not type(i)==float]

topic_list = []
for i in topic_list_raw:
    # special conditions for not clean data
    if "STS" in i:
        i = i.replace("(STS)", " ")
        i = i.replace("(HCI)", " ")
        i = i.replace("(EM)", " ")
        i = i.replace("(CA)", " ")
    if "research interests include" in i:
        i = "poverty, family process, adolescent development and wellbeing, and parent education"
    if "I am working in the " in i:
        i = "Urban sociology, late-modernity, mobilities, precarity, uncertainty, being, experience, ethnography"
    if "No updates" in i:
        continue
    if "Professor Shek" in i:
        i = "positive youth development, family process, scale development, quality of life, programme evaluation, addiction and spirituality"
    key_words = re.split(r"/|,|;|\n|:|, and|including|\(|\)", i)
    key_words = [j.strip() for j in key_words]
    topic_list.extend(key_words)
    
topic_list_v2 = []
for i in topic_list:
    if "-" in i[:2]:
        i = i.replace("-", "")
    if "and" in i[:3]:
        i = i.replace("and", "")
    if "●" in i[:2]:
        i = i.replace("●", "")
    if "." in i[-2:]:
        i = i.replace(".", "")
    if "etc" in i[:4]:
        i = i.replace("etc", "")
    i = i.strip()
    if len(i) >= 2:
        topic_list_v2.append(i)



############# step 2 word to embedding and dimensionality reduction
# # option 1: voyageai
# topic_list = topic_list_v2.copy()
# # pip install -U voyageai
# import voyageai
# vo = voyageai.Client(api_key="") # voyageai api
# initial_list = []
# return_list = []
# for i in range(len(topic_list)):
#     n = i%127
#     initial_list.append(topic_list[i])
#     if (n == 0 and i != 0) or i == len(topic_list)-1:
#         return_list.append(initial_list.copy())
#         initial_list = []
# embedding_result_list = []
# for result in return_list:
#     embedding_result = vo.embed(result, model="voyage-lite-02-instruct", input_type="document")
#     embedding_list = embedding_result.embeddings
#     embedding_result_list.extend(embedding_list)
# embedding_result_list_array = np.array(embedding_result_list)
# dict_df = {'topic_list': topic_list, 'embedding': embedding_result_list} 
# df_embedding = pd.DataFrame(dict_df)
# with pd.ExcelWriter("D:/research topic word cloud/df_embedding.xlsx", engine="openpyxl", mode='w') as writer:
#     df_embedding.to_excel(writer, sheet_name='sheet1', startcol=0, index=False)

# # option 2: Salesforce/SFR-Embedding-Mistral
topic_list = topic_list_v2.copy()
from sentence_transformers import SentenceTransformer
    # Pre-calculate embeddings
# embedding_model = SentenceTransformer("all-mpnet-base-v2")
embedding_model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")

# embeddings = embedding_model.encode(abstracts, show_progress_bar=True) # 这一行放在下面一起
# 下面是两种降维方法：UMAP 和 跳过降维
from umap import UMAP
umap_model = UMAP(n_neighbors=7, n_components=150, min_dist=0.0, metric='cosine', low_memory=True) # , random_state=42
    # 12， 180， 0.0
# from bertopic import BERTopic
# from bertopic.dimensionality import BaseDimensionalityReduction
# # Fit BERTopic without actually performing any dimensionality reduction
# empty_dimensionality_model = BaseDimensionalityReduction()
# topic_model = BERTopic(umap_model=empty_dimensionality_model)
# Controlling Number of Topics
from hdbscan import HDBSCAN
hdbscan_model = HDBSCAN(
    min_cluster_size=2, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

# representation: keyBERT and LLM and ZeroShotClassification
# option 1:
# # 这里先用 keyBERT 作为最后一层，而不是Llama2
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic
representation_model = KeyBERTInspired()

# # option 2:
# # 用 HuggingFaceH4/zephyr-7b-alpha
# # pip install ctransformers[cuda]
# # pip install --upgrade git+https://github.com/huggingface/transformers
# from bertopic import BERTopic
# from ctransformers import AutoModelForCausalLM
# from transformers import AutoTokenizer, pipeline
# # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
# model = AutoModelForCausalLM.from_pretrained(
#     "TheBloke/zephyr-7B-alpha-GGUF",
#     model_file="zephyr-7b-alpha.Q4_K_M.gguf",
#     model_type="mistral",
#     gpu_layers=50,
#     hf=True
# )
# tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
# # Pipeline
# generator = pipeline(
#     model=model, tokenizer=tokenizer,
#     task='text-generation',
#     max_new_tokens=50,
#     repetition_penalty=1.1
# )
# # prompt = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
# # <|user|>
# # I have a research topic that contains the following documents:
# # [DOCUMENTS]

# # The topic is described by the following keywords: '[KEYWORDS]'.

# # Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.</s>
# # <|assistant|>"""
# prompt = """<|system|>You are a helpful, respectful and honest assistant for summarize topics.</s>
# <|user|>
# I have a topic that contains the following documents:
# [DOCUMENTS] 
# [KEYWORDS].

# Please use one to three words to summarize the most important topic from the following documents. Make sure you only return 1 to 3 words and nothing more.</s>
# <|assistant|>"""


# from bertopic.representation import TextGeneration
# # Text generation with Zephyr
# zephyr = TextGeneration(generator, prompt=prompt)
# representation_model = {"Zephyr": zephyr}

# # option 3:
# from bertopic.representation import ZeroShotClassification
# from bertopic import BERTopic
# candidate_topics = ["space and nasa", "bicycles", "sports"]
# representation_model = ZeroShotClassification(candidate_topics, model="facebook/bart-large-mnli")


topic_model = BERTopic(
  # Pipeline models
  embedding_model=embedding_model,
  umap_model=umap_model, #     empty_dimensionality_model
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  representation_model=representation_model,
  # Hyperparameters
  top_n_words=10,   # 默认10，标题一般没那么多字数的，这个参数是不是要缩小？
  verbose=True
)

embeddings = embedding_model.encode(topic_list, show_progress_bar=True) # 这一步每次调整跑首次就可以了

topics1, probs1 = topic_model.fit_transform(topic_list, embeddings)
topic_info1 = topic_model.get_topic_info()
document_info1 = topic_model.get_document_info(topic_list)

with pd.ExcelWriter("D:/research topic word cloud/df_document_info1.xlsx", engine="openpyxl", mode='w') as writer:
    document_info1.to_excel(writer, sheet_name='sheet1', startcol=0, index=False)
with pd.ExcelWriter("D:/research topic word cloud/df_topic_info1.xlsx", engine="openpyxl", mode='w') as writer:
    topic_info1.to_excel(writer, sheet_name='sheet1', startcol=0, index=False)


#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试
#到这里就cluster结束，需要调参，根据结果多次尝试




####################################################################
# step 3: generate bettor words for wordcloud
# """
# Here is a list of topics: "Housing in the ageing society, Aging Society, Memory aging, Ageing, Active Ageing, Ageing and Social Policy, Care Planning, Family Gerontology, aging, elder sexuality, aging experience of LGBT, aging, Mental health in old age, Long-term care, Gerontological care, Lifestyle care, Eye care". Please use one to three words to summarize the most important topic from the list of topics.
# """
# from ctransformers import AutoModelForCausalLM

df_document_info1_excel_name = "D:/research topic word cloud/df_document_info1.xlsx"
df = pd.read_excel(df_document_info1_excel_name)
llmtopic_list = []
promt_text_list = []
documnet_text_list = []
for i in range(len(df)):
    topic = df.loc[i, 'Topic']
    if topic not in llmtopic_list:
        llmtopic_list.append(topic)
        df_with_same_topic = df.loc[df['Topic'] == topic]
        documnet_list = df_with_same_topic['Document'].to_list()
        # promt_text = "Here is a list of topics: " + "\'" + ', '.join(documnet_list)  + "\'" + ". Please use within three words to represent the most important topic from the list of topics."
        promt_text = "Please use within three words to represent the most important topic from the following list of topics: " + "\'" + ', '.join(documnet_list)  + "\'. Just give me the answer."
        promt_text_list.append(promt_text)
        documnet_text_list.append(documnet_list)        
df_promot = pd.DataFrame({'Topic':llmtopic_list, 
                          'documnet_list':documnet_text_list,
                          'promt_text':promt_text_list})
df_promot['llm_return'] = ''
for i in range(len(df_promot)):
    promt_text = df_promot.loc[i, 'promt_text']
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # https://huggingface.co/google/gemma-2b-it
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", token='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')
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", token='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')
    input_ids = tokenizer(promt_text, return_tensors="pt")
    outputs = model.generate(**input_ids, max_new_tokens=100)
    llm_return = tokenizer.decode(outputs[0])
    df_promot.loc[i, 'llm_return'] = llm_return
with pd.ExcelWriter("D:/research topic word cloud/df_llm_topic_representation.xlsx", engine="openpyxl", mode='w') as writer:
    df_promot.to_excel(writer, sheet_name='sheet1', startcol=0, index=False)


####################################################################
# step 4: clean the llm generated topics
df_promot_name = "D:/research topic word cloud/df_llm_topic_representation.xlsx"
df_promot = pd.read_excel(df_promot_name)
df_promot['clean_topic'] = ''
for i in range(len(df_promot)):
    llm_return = df_promot.loc[i, 'llm_return']
    
    if "Answer:" in llm_return:
        clean_topic = llm_return.split("Answer:")[-1].split("**")[0].strip()
    elif "from the list is **" in llm_return:
        clean_topic = llm_return.split("from the list is **")[-1].split("**")[0].strip()
    elif "the answer" in llm_return and "**" in llm_return:
        clean_topic = re.search(r'\*\*(.*?)\*\*', llm_return).group(1)
        if len(clean_topic) > 30:
            clean_topic =  re.findall(r'\*\*(.*?)\*\*', llm_return)[-1]
    elif "**" in llm_return:
        clean_topic = re.search(r'\*\*(.*?)\*\*', llm_return).group(1)
    elif "the answer.\n\n" in llm_return:
        clean_topic = llm_return.split("the answer.\n\n")[-1].split(".")[0].strip()
        if "The answer is" in clean_topic:
            clean_topic = clean_topic.split("The answer is")[-1]
            if "\'" in clean_topic:
                clean_topic =  re.search(r"\'(.*?)\'", clean_topic).group(1)
            if "\"" in clean_topic:
                clean_topic =  re.search(r"\"(.*?)\"", clean_topic).group(1)
        if "is the most" in clean_topic:
            clean_topic = clean_topic.split("is the most")[0].strip()
    else:
        print(llm_return)
        clean_topic = ''
    df_promot.loc[i, 'clean_topic'] = clean_topic

df_document_info1_excel_name = "D:/research topic word cloud/df_document_info1.xlsx"
df = pd.read_excel(df_document_info1_excel_name)
df['clean_topic'] = ''
for i in range(len(df)):
    topic = df.loc[i, 'Topic']
    if topic == -1:
        df_with_same_topic = df.loc[df['Topic'] == topic]
        documnet_list = df_with_same_topic['Document'].to_list()
        df.loc[i, 'clean_topic'] = str(documnet_list)
    else:
        df.loc[i, 'clean_topic'] = df_promot[df_promot['Topic'] == topic]['clean_topic'].to_list()[0]
with pd.ExcelWriter("D:/research topic word cloud/df_with_clean_topic.xlsx", engine="openpyxl", mode='w') as writer:
    df.to_excel(writer, sheet_name='sheet1', startcol=0, index=False)


########################
# step 5: Generate the word cloud
# step 5.1: get topic_list
topic_list = []
df_with_clean_topic_name = "D:/research topic word cloud/df_with_clean_topic.xlsx"
df = pd.read_excel(df_with_clean_topic_name)
for i in range(len(df)):
    topic = df.loc[i, 'Topic']
    clean_topic = df.loc[i, 'clean_topic']
    if topic != -1:
        topic_list.append(clean_topic)
import ast
topic_list.extend(ast.literal_eval(df[df['Topic'] == -1]['clean_topic'].sample(n=1).to_list()[0]))

# step 5.2: wordcloud
from PIL import Image #to load our image
custom_mask = np.array(Image.open('D:/research topic word cloud/cloud_highresolution.png')) 
wordcloud = WordCloud(width=4000, height=2000, 
                      background_color='white',
                      colormap='RdYlGn', # matplotlib colormap
                      max_words=200,
                      min_font_size=6,
                      mode='RGB',
                      relative_scaling=0.5,
                      mask=custom_mask
                      ).generate(' '.join(topic_list))
plt.figure(figsize=(60, 30))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
from datetime import datetime
current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
wordcloud.to_file('D:/research topic word cloud/'+current_datetime+'.png')



