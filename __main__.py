from Model import Model
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from datetime import datetime
import os


def load_dataset():
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # 设置数据集路径
    data_dir = 'dataset'

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 尝试下载数据集
    try:
        train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        return train_set, test_set
    except Exception as e:
        print(f"Error occurred: {e}")
    return None, None


def train_model(model: Model, train_loader: DataLoader):
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    for i in range(config.TRAINING_SET['num_epochs']):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch: {i} / {config.TRAINING_SET['num_epochs'] - 1}, loss: {running_loss / len(train_loader)}")

    text = f"models/{input()}.pth"
    print(text)
    torch.save(model, text)

    print("training finished")
    return


def main():
    model = Model()
    train_set, test_set = load_dataset()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    mode = eval(input())
    if mode == 1:
        # train mode
        train_model(model, train_loader)

    elif mode == 2:
        # test mode
        test_model(model, test_loader)

    else:
        print("wrong mode")


def test_model(model: Model, test_loader: DataLoader):
    # 进行测试
    model = torch.load(f'models/{input()}.pth', weights_only=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config.TRAINING_SET['device']), labels.to(config.TRAINING_SET['device'])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(correct, total, correct/total)


if __name__ == "__main__":
    main()
