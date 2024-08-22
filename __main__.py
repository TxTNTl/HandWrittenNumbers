from Model import Model
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
import os


def load_dataset(mode):
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # 设置数据集路径
    data_dir = 'dataset'

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 建议在设置目录预先下载好数据集，root代表路径，train代表是训练还是测试，download代表不存在数据集是否下载，transform代表输入进来之后如何变化
    if mode == 'train':
        try:
            train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
            return train_set
        except Exception as e:
            print(f"Error occurred: {e}")

    elif mode == 'test':
        try:
            test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
            return test_set
        except Exception as e:
            print(f"Error occurred: {e}")
    else:
        print("Invalid mode")
        return None, None


def train_model():
    model = Model()
    train_set = load_dataset('train')
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 开启训练模式
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

    print("Finished Training")
    print("The name for saving model?")
    text = f"models/{input()}.pth"
    print(text)
    torch.save(model, text)
    print("training finished")
    return


def test_model():
    test_set = load_dataset('test')
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    print("Please input the name of the model")
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


def main():
    print("Choose your mode. 1 for training or 2 for testing mode")
    mode = eval(input())
    if mode == 1:
        # train mode
        train_model()

    elif mode == 2:
        # test mode
        test_model()

    else:
        print("wrong mode")


if __name__ == "__main__":
    main()
