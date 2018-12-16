import argparse
import pandas as pd
import numpy as np
import re
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# set parameters
ap = argparse.ArgumentParser()
ap.add_argument("--m", required=True, type=str)
args = vars(ap.parse_args())

# read json
run_model = args["m"]
print("running model:", run_model)
if run_model == "local":
    config_path = "/Users/haoran/Documents/BUAA_course/matchine_learnign/final_work/team_work/Sentiment_Analysis/" \
                  "config/local_config.json"
else:
    config_path = "/home/poac/code_mac/Sentiment_Analysis/config/server_config.json"
config = json.load(open(config_path))

# 将标签数据和未标签数据合并

# unlabeled
path1 = config["Dir"] + config["unlabeled_txt_path"]
unlabel = []
with open(path1, 'r') as file:
    while(True):
        line = file.readline()
        if not line:
            print("file over")
            break
        line1 = line.replace("	", ",").replace("\n", "").split(",")
        unlabel.append([line1])
print("unlabeled shape:", np.array(unlabel).shape)


# labeled
path2 = config["Dir"] + config["labeled_txt_path"]
label = []
with open(path2, 'r') as file:
    while(True):
        line2 = file.readline()
        if not line2:
            print("file over")
            break
        line3 = line2.replace("	", ",").replace("\n", "").split(",")
        label.append([line3])
out_labeled = label   # 只包含有标签的数据，用来训练分类器
print("labeled shape:", np.array(label).shape)


# test data
path4 = config["Dir"] + config["test_txt_path"]
test_data = []
with open(path4, 'r') as file:
    while(True):
        line5 = file.readline()
        if not line5:
            print("file over")
            break
        line6 = line5.replace("	", ",").replace("\n", "").split(",")
        test_data.append([line6])


# 合并
out = pd.concat([pd.DataFrame(label), pd.DataFrame(unlabel)], axis=0)
out = list(np.array(out))

out_all = []   # 包含有标签数据和无标签数据，用来训练自编码器
for i in range(len(out)):
    tem = re.sub(r'\'', "", str(out[i]))
    tem = re.sub(r'\"', "", tem)
    tem = re.sub(r'\[', "", tem)
    tem = re.sub(r'\)', "", tem)
    tem = re.sub(r'\]', "", tem).strip().replace("  ", "").replace("list(", "").split(",")
    out_all.append(str(tem))

print(np.array(out_all).shape)

# 清理test
out_test = []
for i in range(len(test_data)):
    tem1 = re.sub(r'\'', "", str(test_data[i]))
    tem1 = re.sub(r'\"', "", tem1)
    tem1 = re.sub(r'\[', "", tem1)
    tem1 = re.sub(r'\)', "", tem1)
    tem1 = re.sub(r'\]', "", tem1).strip().replace("  ", "").replace("list(", "").split(",")
    out_test.append(str(tem1))

# 计算每个词的TF-IDF
vectorizer = CountVectorizer(stop_words='english', lowercase=True, min_df=0.005)
X = vectorizer.fit_transform(out_all)
word = vectorizer.get_feature_names()
print("文本关键词：", word)
print("词频结果：", X.toarray())

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
tfidf_value = tfidf.toarray()   # tfidf_value[i][j]，i表示第i个句子，j表示第j个词

tfidf_labeled = tfidf_value[0:20000, :]

print("tfidf shape：", tfidf_value.shape)

tfidf_test = transformer.transform(vectorizer.transform(out_test)).toarray()

print("tfidf_test shape:", np.array(tfidf_test).shape)

# 将TF-IDF值写入文件
wd_all = pd.DataFrame(tfidf_value)
wd_all.to_csv(config["Dir"]+config["tfidf_value"], header=None, index=None)

wd_labeled = pd.DataFrame(tfidf_labeled)
wd_labeled.to_csv(config["Dir"]+config["tfidf_labeled"], header=None, index=None)

wd_test = pd.DataFrame(tfidf_test)
wd_test.to_csv(config["Dir"]+config["tfidf_test"], header=None, index=None)



