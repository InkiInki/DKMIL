# DKMILRepertoire
This repository holds the Pytorch implementation for the method described in the paper:
> Data-Driven Knowledge Fusion for Deep Multi-Instance Learning
Yu-Xuan Zhang, Zhengchun Zhou, Xingxing He, Avik Ranjan Adhikary, Bapi Dutta.
IEEE Transactions on Neural Networks and Learning Systems (2024)

![](DKMIL.png)


# Data Preparation
We use two datasets in our paper for demonstration: 
1) [Benchmark dataset](https://www.kaggle.com/inkiyinji) 
2) [Image dataset](dataset/bag_generator2D.py).

# Training:
For benchmark dataset:

    python main.py

For image dataset (MNIST, CIFAR10, STL10):

    python main2D.py

All rights reserved.

# Citation

    @article{Zhang:2024:115:data,
    author		=	{Zhang, Yu-Xuan and Zhou, Zhengchun and He Xingxing and Adhikary, Avik Ranjan and Dutta, Bapi},
    title		=	{Data-driven knowledge fusion for deep multi-instance learning},
    journal		=	{{IEEE} Transactions on Neural Networks and Learning Systems},
    year		=	{2023},
    pages		=	{1--15},
    }
