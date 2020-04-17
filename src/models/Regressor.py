from src.base.base_net import BaseNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class Regressor(BaseNet):

    def __init__(self):
        super().__init__('regressor')
        self.fc = nn.Linear(3,20)
        self.fc_1 = nn.Linear(20,20)
        self.fc_2 = nn.Linear(20, 20)
        self.fc_3 = nn.Linear(20, 1)
        self.activation = F.leaky_relu
        pass

    def forward(self, x):
        x = self.activation(self.fc(x))
        x = self.activation(self.fc_1(x))
        x = self.activation(self.fc_2(x))
        x = self.fc_3(x)
        return x
