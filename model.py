import torch
from torch import nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# optimizer_fn = lambda x: optim.Adam(x, lr=1e-5)
optimizer_fn = lambda x: optim.SGD(x, lr=1e-3, )


# 可以通过这段代码了解输入图像的维度信息
# torch.Size([1, 1, 28, 28]) 分别对应BCHW
# data=datasets.EMNIST('.',split="byclass",transform=transforms.Compose([transforms.ToTensor()]))
# dataloader=DataLoader(data,)
# image,_=next(iter(dataloader))
# print(image.shape)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            # 1*28*28 -> 32*26*26
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            # 32*26*26 -> 32*24*24
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.LeakyReLU(),
            # 32*24*24 -> 32*12*12
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(32),
            # 32*12*12 -> 64*10*10
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            # 64*10*10 -> 64*8*8
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.LeakyReLU(),
            # 64*8*8 -> 64*4*4
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.BatchNorm1d(64 * 4 * 4),
            nn.Linear(64 * 4 * 4, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 62),
        )

        self.net2 = nn.Sequential(
            # 1*28*28 -> 32*26*26
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(4),
            # 32*26*26 -> 32*24*24

            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Flatten(),
            nn.BatchNorm1d(2 * 4 * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2 * 4 * 4, 62),
        )

        self.net3 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 62),
        )

    def forward(self, x):
        return self.net(x)
