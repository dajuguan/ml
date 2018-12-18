"#_*_ coding:utf-8 _*_"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))
import argparse
import torchvision
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import json
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
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
    from model.encoder_lstm import autoencoder_lstm
    config_path = "/Users/haoran/Documents/BUAA_course/matchine_learnign/final_work/team_work/Sentiment_Analysis/" \
                  "config/local_config.json"
else:
    from home.poac.code_mac.Sentiment_Analysis.model.autoencoder_lstm import encoder_lstm
    config_path = "/home/poac/code_mac/Sentiment_Analysis/config/server_config.json"
config = json.load(open(config_path))

# 读取训练数据集，及标签
all_data = pd.read_csv(config["Dir"]+config["tfidf_value"], header=None)
labeled_data = pd.read_csv(config["Dir"]+config["tfidf_labeled"], header=None)
label = pd.read_csv(config["Dir"]+config["label"], header=None)
print("all data shape", np.array(all_data).shape)
print("labeled data shape:", np.array(labeled_data).shape)
print("label shape:", np.array(label).shape)

# 训练自编码器，使用all_data
# 定义超参数
batch_size = 50
learning_rate = 1e-3
num_epoches = 20
seq_num = 1

# 将数据转换成torch
label_all = torch.ones(np.array(all_data).shape[0])
all_data = torch.from_numpy(np.array(all_data)).float()
torch_dataset = Data.TensorDataset(all_data, label_all)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

# 定义自编码器模型
in_dim, hidden_dim, n_layer, n_out = 2919, 150, 1, 2919
model_encoder_decoder = autoencoder_lstm(in_dim, hidden_dim, n_layer, n_out)

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss(reduce=True, size_average=True)
optimizer = optim.Adam(model_encoder_decoder.parameters(), lr=learning_rate)

# 开始训练自编码器
LOSS = []
for epoch in range(num_epoches):
    for step, (batch_x, _) in enumerate(loader):
        start = time.clock()
        num_batch = batch_x.size(0)  # 根据数据动态控制batch_size
        x = Variable(batch_x)
        input = x.view(num_batch, seq_num, in_dim)
        out, _ = model_encoder_decoder(input)
        loss = mse(x, out)
        LOSS.append(loss.data[0])

        # 梯度归零，反向传播，更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.clock()

        if (step+1) % 50 == 0:
            print('Epoch [{}/{}], loss: {}, time: {}'.format(epoch, num_epoches, loss.data[0], (end - start)))

# 自编码器训练完成
# 保存模型
torch.save(model_encoder_decoder, config["Dir"]+config["model_save"])  # 保存整个网络

# 统计训练损失
LOSS = np.array(LOSS)
plt.figure(figsize=(10, 10))
plt.plot(LOSS, label='train loss')
plt.legend()
plt.savefig(config["Dir"]+config["loss_figure"])
# plt.show()







