'''
该文件的作用就是将tiger_game的数据集进行随机划分
'''
import csv
import os
import random
import pandas as pd
dataset_path = r'YOUR_DIR/tissue-cells/images'
image_name = os.listdir(dataset_path)  # 获取到所有的图片的名字
print(len(image_name))

train_list = []             # 创建训练集列表
valid_list = []             # 创建验证集列表
test_list = []



############################ 划分训练集
sample_num = 1316
train_sample = random.sample(image_name,sample_num)
train_list.extend(train_sample)

# 保存数据集到csv文件中
train = pd.DataFrame(data=train_list)
train.to_csv('train.csv',columns=None)
print("训练集的长度：",len(train_list))



############################ 划分验证集
sample_num_valid = 376
test_valid_list = [x for x in image_name if x not in train_list]
valid_sample = random.sample(test_valid_list,sample_num_valid)
valid_list.extend(valid_sample)

# 保存数据集到csv文件中
valid = pd.DataFrame(data=valid_list)
valid.to_csv('valid.csv',columns=None)
print("验证集的长度：",len(valid_list))






############################ 划分测试集
sample_num_valid = 189
test_list = [x for x in test_valid_list if x not in valid_list]
print("测试集的长度：",len(test_list))
# 保存数据集到csv文件中
test = pd.DataFrame(data=test_list)
test.to_csv('test.csv',columns=None)






# 验证 三个数据集里面有没有重复的
repeat = [x for x in image_name if x in train_list and x in valid_list and x in test_list]
print("重读的元素数量",len(repeat))





