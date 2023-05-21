import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# 加载ResNet50模型
resnet50 = models.resnet50(pretrained=True)
# 截断模型最后一层
model = nn.Sequential(*list(resnet50.children())[:-1])

# CIFAR-10中的10个类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 加载CIFAR-10测试数据
transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# 提取所有测试样本的特征，并将其存储在numpy数组中
features = []
labels = []
for i, (inputs, targets) in enumerate(testloader):
    features.append(model(inputs).detach().numpy().squeeze())
    labels.append(classes[targets])
features = np.array(features)

# 对特征进行k-means聚类
kmeans = KMeans(n_clusters=10, random_state=0).fit(features)

# 打印每个聚类的中心点
print(kmeans.cluster_centers_)

# 确定每个图像的类别
predictions = kmeans.predict(features)

# 将类别标签和对应的图像保存到磁盘
for i in range(len(predictions)):
    img = Image.fromarray(np.uint8(testset[i][0].numpy().transpose((1, 2, 0)) * 255))
    img.save(f"{classes[predictions[i]]}/{i}.png")
