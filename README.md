# Single Digit Character Recognition Based on MNIST Dataset

## Project Introduction

This project is a small beginner-level exercise for deep learning, 
aimed at helping familiarize with the PyTorch package. 
The final recognition accuracy can reach 97.6%. 
Of course, there is still room for improvement in accuracy, 
which can be achieved by choosing different loss functions, optimizers, increasing the number of iterations, changing the learning rate, etc. 
The level of this project is meant to be a practice exercise, 
so there is no need to elaborate further.

## File Introduction

The main content is in the __main__.py and Model.py files. 
The config.py file is intended to configure all the configurable options, 
but the project doesn’t require that many. 
Therefore, only the number of iterations is configurable. 
The selected model consists of two fully connected layers, 
which is relatively simple. The only thing worth noting is the dataset. 
I was unable to download it using PyTorch’s download feature because the local network could not access the website. 
Therefore, it is recommended to download it beforehand from the internet.



# 基于MNIST数据集的单个数字字符识别

## 项目介绍
本项目仅作为深度学习的入门的练手的小项目，帮助熟悉Pytorch包，最终识别率能够达到97.6%。
当然，准确率有一定地可提升空间，可以选择不同的损失函数、优化器，提高迭代次数，改变学习率等等。
本项目的水平仅可以作为一个练手，因此不过多赘述

## 文件介绍
主要内容在__main__.py和Model.py文件中，config.py文件本意是进行所有可配置项的配置的，但是该项目用不到那么多。
所以只有一个迭代次数用来修改。至于选择的模型为两个全连接层，较为简单。唯一值得说明的是数据集，
我是用Pytorch的下载功能无法下载，因为本地打不开网页，因此建议网上提前下好。