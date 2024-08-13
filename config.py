import torch

# 配置文件

IMAGE_SET = {
    'input_size' : 28 * 28
}

TRAINING_SET = {
    'num_epochs' : 10,
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
}