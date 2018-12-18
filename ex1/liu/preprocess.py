# 创建清理文件存放目录
import os

import re
from bs4 import BeautifulSoup as Bs
import pandas as pd
import numpy as np
from collections import Counter
import h5py

path1 = 'cleaned/'
exists1 = os.path.exists(path1)
if not exists1:
    os.makedirs(path1)
path2 = 'cleaned_with_top_4k_used/'
exists2 = os.path.exists(path2)
if not exists2:
    os.makedirs(path2)

# 原始数据路径
path_labeled_train_data = 'sentiment/labeledTrainData.tsv'
path_unlabeled_train_data = 'sentiment/unlabeledTrainData.tsv'
path_test_data = 'sentiment/testData.tsv'

# 清理后文件存放路径
path_cleaned_labeled_train_data = 'cleaned/labeledTrainData.tsv'
path_cleaned_unlabeled_train_data = 'cleaned/unlabeledTrainData.tsv'
path_cleaned_test_data = 'cleaned/testData.tsv'
rge = re.compile(r'\b[a-z0-9\']+\b', flags=re.IGNORECASE)

# clean labeled train data
labeledTrainData = []
dataset1 = pd.read_csv(path_labeled_train_data, sep='\t', header=0,
                       error_bad_lines=False, encoding='ISO-8859-1')
for item in dataset1['review']:
    item = Bs(item, 'lxml').get_text().lower()
    item = ' '.join(rge.findall(item))
    labeledTrainData.append(item)

labeledTrainDataFrame = pd.DataFrame([dataset1['id'], dataset1['sentiment'],
                                      pd.Series(labeledTrainData)],
                                     index=['id', 'sentiment', 'review']).transpose()
labeledTrainDataFrame.to_csv(path_cleaned_labeled_train_data, sep='\t', index=False)

# clean unlabeled train data
unlabeledTrainData = []
dataset2 = pd.read_csv(path_unlabeled_train_data, sep='\t', header=0,
                       error_bad_lines=False, encoding='ISO-8859-1')
for item in dataset2['review']:
    item = Bs(item, 'lxml').get_text().lower()
    item = ' '.join(rge.findall(item))
    unlabeledTrainData.append(item)

unlabeledTrainDataFrame = pd.DataFrame([dataset2['id'], pd.Series(unlabeledTrainData)],
                                       index=['id', 'review']).transpose()
unlabeledTrainDataFrame.to_csv(path_cleaned_unlabeled_train_data, sep='\t', index=False)
# clean test data
testData = []
dataset3 = pd.read_csv(path_test_data, sep='\t', header=0,
                       error_bad_lines=False, encoding='ISO-8859-1')
for item in dataset3['review']:
    item = Bs(item, 'lxml').get_text().lower()
    item = ' '.join(rge.findall(item))
    testData.append(item)

testDataFrame = pd.DataFrame([dataset3['id'], pd.Series(testData)],
                             index=['id', 'review']).transpose()
testDataFrame.to_csv(path_cleaned_test_data, sep='\t', index=False)

# 获取词频前4k的词
total_words = []
for item in labeledTrainData:
    total_words.extend(item.split(' '))
for item in unlabeledTrainData:
    total_words.extend(item.split(' '))
for item in testData:
    total_words.extend(item.split(' '))
word = sorted(Counter(total_words).items(), key=lambda x: x[1], reverse=True)[:4000]
iden = list(range(4000))
df = pd.concat([pd.DataFrame(word, columns=['word', 'count']),
                pd.DataFrame(iden, columns=['id'])], axis=1)
df.to_csv(path1 + 'top_4k_used.csv', sep='\t', header=True)
top_4k_used_words = df['word'].values
top_4k_used_ids = df['id'].values + 1
word_dict = dict(zip(top_4k_used_words, top_4k_used_ids))

path_clean_labeled_train_data_top_5k_used = path2 + 'labeledTrainData.tsv'
path_clean_unlabeled_train_data_top_5k_used = path2 + 'unlabeledTrainData.tsv'
path_clean_test_data_top_5k_used = path2 + 'testData.tsv'

path_vec = path2 + 'data.h5'
h = h5py.File(path_vec)
# 去掉labeledTrainData不在top_5k_used里面的词,将每个句子截断或延长为252的长度
# 首尾为-1，不足的补上0
labeledTrainData = np.zeros((20000, 252), dtype=np.int16)
dataset1 = pd.read_csv(path_cleaned_labeled_train_data, sep='\t', header=0)
for i, item in enumerate(dataset1['review']):
    temp = np.zeros(252, dtype=np.int16)
    temp[0], temp[-1] = 4001, 4002
    item = [word_dict[word] for word in item.split(' ') if word in top_4k_used_words]
    if len(item) < 250:
        temp[1:len(item) + 1] = item
    else:
        temp[1:-1] = item[:250]
    labeledTrainData[i] = temp
h.create_dataset('labeledTrainData', data=labeledTrainData)
print('labeled train data are cleaned', labeledTrainData.shape)

# 去掉unlabeledTrainData不在top_5k_used里面的词，将每个句子截断或延长为252的长度
# 首尾为-1，不足的补上0
unlabeledTrainData = np.zeros((49998, 252), dtype=np.int16)
dataset2 = pd.read_csv(path_cleaned_unlabeled_train_data, sep='\t', header=0)
for i, item in enumerate(dataset2['review']):
    temp = np.zeros(252, dtype=np.int16)
    temp[0], temp[-1] = 4001, 4002
    item = [word_dict[word] for word in item.split(' ') if word in top_4k_used_words]
    if len(item) < 250:
        temp[1:len(item) + 1] = item
    else:
        temp[1:-1] = item[:250]
    unlabeledTrainData[i] = temp
h.create_dataset('unlabeledTrainData', data=unlabeledTrainData)
print('unlabeled train data are cleaned', unlabeledTrainData.shape)

# 去掉testData不在top_5k_used里面的词，将每个句子截断或延长为252的长度
# 首尾为-1，不足的补上0
testData = np.zeros((5000, 252), dtype=np.int16)
dataset3 = pd.read_csv(path_cleaned_test_data, sep='\t', header=0)
for i, item in enumerate(dataset3['review']):
    temp = np.zeros(252, dtype=np.int16)
    temp[0], temp[-1] = 4001, 4002
    item = [word_dict[word] for word in item.split(' ') if word in top_4k_used_words]
    if len(item) < 250:
        temp[1:len(item) + 1] = item
    else:
        temp[1:-1] = item[:250]
    testData[i] = temp
h.create_dataset('testData', data=testData)
print('test data are cleaned', testData.shape)
