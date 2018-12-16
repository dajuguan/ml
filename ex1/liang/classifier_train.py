import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))
import torch
import argparse
import numpy as np
import pandas as pd
import json
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

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

# 获取测试集id
id_data = pd.read_csv(config["Dir"]+config["test_csv_path"])
id = id_data["id"]

# 读取有标签数据，用来训练分类器
labeled_data = pd.read_csv(config["Dir"]+config["tfidf_labeled"], header=None)
label = pd.read_csv(config["Dir"]+config["label"], header=None)
print(np.array(label).shape)

# 导入自编码模型
encoder_model = torch.load(config["Dir"]+config["model_save"])

encoder_model.eval()
labeled_data = torch.from_numpy(np.array(labeled_data)).float()
labeled_data = Variable(labeled_data).view(-1, 1, 2919)

_, inter_data = encoder_model(labeled_data)
inter_data = inter_data.data.numpy()
print("inter_data shape:", inter_data.shape)

# 划分数据集
train_x, validation_x, train_y, validation_y = train_test_split(np.array(inter_data), np.array(label),
                                                                 shuffle=True,
                                                                 test_size=0.3)

# 训练模型
model = RandomForestClassifier(n_estimators=500)
# model = xgb.XGBClassifier()
print("train_y shape:", train_y)
model.fit(train_x, train_y)

# 验证模型
validation_pred = model.predict(validation_x)
accuracy = accuracy_score(validation_y, np.array(validation_pred))
print(accuracy)

# 测试集
test = pd.read_csv(config["Dir"]+config["tfidf_test"], header=None)
test = torch.from_numpy(np.array(test)).float()
test = Variable(test).view(-1, 1, 2919)
_, inter_test = encoder_model(test)
inter_test = inter_test.data.numpy()
pred_test = model.predict(inter_test)

print("pred shape:", np.array(pred_test).shape)
print("id shape:", np.array(id).shape)
result = pd.concat([pd.DataFrame(id), pd.DataFrame({'sentiment': pred_test})], axis=1)
wd = pd.DataFrame(result)
wd.to_csv(config["Dir"]+config["submission"], index=None)

