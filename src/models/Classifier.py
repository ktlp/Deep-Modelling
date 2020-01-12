from src.base.base_net import BaseNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(BaseNet):

    def __init__(self):
        super().__init__('classifier')
        self.fc = nn.Linear(3,100)
        self.fc_1 = nn.Linear(100, 100)
        self.fc_2 = nn.Linear(100, 4)
        self.output = nn.LogSoftmax()
        pass

    def forward(self, x):
        x = self.fc(x)
        x = self.fc_1(x)
        x = self.output(self.fc_2(x))
        return x
