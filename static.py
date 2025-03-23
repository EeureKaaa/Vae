import torch
from torchvision import datasets, transforms
from collections import Counter

# 加载 MNIST 训练集
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

# 获取所有标签
labels = torch.tensor(mnist_train.targets)

# 统计每个数字的数量
count_dict = Counter(labels.numpy())

# 打印结果
for digit in range(10):
    print(f"Digit {digit}: {count_dict[digit]} samples")
