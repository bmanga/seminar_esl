import torch
from torch import nn
import torch.nn.functional as F


class Net (nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 10)

    def forward(self, x):
        x = self.conv1(x)
        # [10, 24, 24]
        x = F.max_pool2d(x, 2)
        # [10, 12, 12]
        x = F.relu(x)

        x = self.conv2(x)
        # [20, 8, 8]
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        # [20, 4, 4]
        x = F.relu(x)
        x = x.view(-1, 320)
        # [320]
        x = self.fc1(x)
        # [50]
        x = F.relu(x)

        # [10]
        return x

