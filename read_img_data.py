import os
import random

# 指定数据集根目录和类别名称
root_dir = "F:\\city_sing"
class_names = ["BQ", "CT", "KG", "LI"]

# 创建txt文件
with open("data.txt", "w") as f:
    # 遍历每个类别文件夹
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)
        # 遍历每个图像文件
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            # 写入txt文件
            f.write("{},{}\n".format(img_path, i))


# 读取并打乱文件
data_file = 'data.txt'

# Read in the data file
with open(data_file, 'r') as f:
    data = f.readlines()

# Shuffle the data
random.shuffle(data)

# Split the data into train and test sets
split_index_1 = int(0.7 * len(data))
split_index_2 = int(0.9 * len(data))
train_data = data[:split_index_1]
val_data = data[split_index_1: split_index_2]
test_data = data[split_index_2: ]

# Write the train and test data to separate files
with open('train.txt', 'w') as f:
    f.writelines(train_data)

with open('val.txt', 'w') as f:
    f.writelines(val_data)

with open('test.txt', 'w') as f:
    f.writelines(test_data)

