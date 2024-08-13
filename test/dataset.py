import os
import torchvision
import torchvision.transforms as transforms

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor()
])

# 设置数据集路径
data_dir = '../dataset'

# 检查数据目录是否存在
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 尝试下载数据集
try:
    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
except Exception as e:
    print(f"Error occurred: {e}")
