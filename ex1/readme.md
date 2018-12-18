情感分析作业

# 主要分为两个文件
- hw4.py是主程序
- utils/util是处理文字的程序

# 训练
> python3 hw4.py semi semi 

# 测试
> python3 hw4.py semi test --load_model semi

How to use
supervised training (for example)
> python3 hw4.py <model_name> train --cell LSTM

semi-supervised training
> python3 hw4.py <semi_model_name> semi --load_model <model_name>