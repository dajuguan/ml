import pandas as pd
import re
from bs4 import BeautifulSoup as bs
import collections


def read_data(path, name='train_labeled'):
    dataset = pd.read_csv(path, sep='\t', header=0, error_bad_lines=False, encoding='ISO-8859-1')
    data = []
    for item in dataset['review']:
        item = bs(item, 'lxml').get_text()
        item = re.findall(r'\b[a-z0-9\']+\b', item, flags=re.IGNORECASE)
        data.append(' '.join(item).lower())
    if name == 'train_labeled':
        labels = [k[1] for k in dataset['sentiment'].items()]
        return data, labels
    else:
        return data


def resave(data, name):
    if name == 'cleanedLabeledTrainData':
        df = pd.DataFrame(data, index=['sentiment', 'review']).transpose()
        df.to_csv('cleaned/' + name + '.tsv', '\t')
    else:
        df = pd.DataFrame(data, columns=['review'])
        df.to_csv('cleaned/' + name + '.tsv', sep='\t')


train_data_labeled, train_labels = read_data('sentiment/labeledTrainData.tsv')
train_data_unlabeled = read_data('sentiment/unlabeledTrainData.tsv', None)
test_data = read_data('sentiment/testData.tsv', None)

for data, name in zip([[train_labels, train_data_labeled], train_data_unlabeled, test_data],
                      ['cleanedLabeledTrainData', 'cleanedUnlabeledTrainData', 'cleanedTestData']):
    resave(data, name)

df_train_labeled = pd.read_csv('cleaned/cleanedLabeledTrainData.tsv', index_col=0, sep='\t', header=0)
df_train_unlabeled = pd.read_csv('cleaned/cleanedUnlabeledTrainData.tsv', index_col=0, sep='\t', header=0)
df_test = pd.read_csv('cleaned/cleanedTestData.tsv', index_col=0, sep='\t', header=0)

total_words = []
for df in [df_train_labeled, df_train_unlabeled, df_test]:
    for item in df['review']:
        total_words.extend(item.split())
word_count = sorted(collections.Counter(total_words).items(), key=lambda x: x[1], reverse=True)[:4000]
pd.DataFrame(word_count).to_csv('top_4k_used.tsv', sep='\t', header=None)
