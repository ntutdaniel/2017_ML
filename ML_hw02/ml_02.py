#encoding:utf-8
import pandas as pd
import string
import nltk
#ltk.download('punkt')
#nltk.download('stopwords')
import scipy
from sklearn.feature_extraction.text import CountVectorizer
import math
import numpy as np
import os


train = pd.read_csv("./train.csv", sep=",")
#print train

text = train.text

#刪除nltk
#去標點符號
exclude = set(string.punctuation)
stopwords = nltk.corpus.stopwords.words('english')
for i in range(len(stopwords)):
    stopwords[i] = stopwords[i].encode('ascii', 'ignore')
#print stopwords

for i in range(len(text)):
    text[i] = ''.join(ch for ch in text[i] if ch not in exclude)
    text[i] = str.lower(text[i])
    text[i] = nltk.word_tokenize(text[i])#to array
    text[i] = [word for word in text[i] if str(word.lower()) not in stopwords]
    #text[i] = ' '.join(ch for ch in text[i])

#train.to_csv('./train_remove_stopwords')

#tf
indenpenden_words = {}
for i in range(len(text)):
    for j in range(len(text[i])):
        if text[i][j] not in indenpenden_words:
            indenpenden_words[text[i][j]] = 1
        else:
            indenpenden_words[text[i][j]] += 1

#idf
train_text_length = len(text)
for word, count in indenpenden_words.items():
    indenpenden_words[word] = count#math.log(float(train_text_length)/count, 10)

qr_c = len(indenpenden_words) / 100

#get feature
features = [word for word, idf in sorted(indenpenden_words.items(),key=lambda x:x[1], reverse=True)[:qr_c]]

#set values
for i in range(len(text)):
    #print text[i]
    values = [0] * qr_c
    for j in range(len(text[i])):
        if text[i][j] in features:
            # if i < 2:
            #     print i, text[i][j]
            index = features.index(text[i][j])
            values[index] += 1
    text[i] = ','.join(str(ch) for ch in values)

#out put csv
out_p = './train_convert.csv'
if os.path.exists(out_p):
    os.remove(out_p)

f = open(out_p, 'w')

#header
features.append('author')
features.insert(0, 'id')
features_str = ','.join(str(ch) for ch in features)
f.writelines(features_str + '\n')

for i in range(len(text)):
    f.writelines(train.id[i] + ',' + text[i] + ',' + train.author[i] + '\n')

f.close()








