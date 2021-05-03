from __future__ import annotations

import torch
from models.base_model import BaseModel

import torch.nn as nn
import torch.optim as optim
from torch.tensor import Tensor
from torch.utils.data.dataset import Dataset


class CIFARModel(BaseModel):

    epochs = 25
    batch_size = 64

    def __init__(self, dataset: Dataset, testset: Dataset):
        super().__init__(dataset, testset)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.linear = nn.Sequential(
            nn.Linear(1024 * 4 * 4 + 256 * 8 * 8, 120),
            nn.ReLU(),
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), momentum=0.9, lr=0.004)
        # self.optimizer = optim.Adagrad(self.parameters(), lr=4e-3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        y: Tensor = self.conv2(x)
        x = self.conv3(y)
        x = x.view(-1, 1024 * 4 * 4)
        y = y.view(-1, 256 * 8 * 8)

        z = torch.cat([x, y], 1)

        z = self.linear(z)

        return z

    def train_model(self) -> float:
        data_count = len(self.dataloader)

        average_loss = 0
        for i, data in enumerate(self.dataloader):
            inputs: Tensor = data[0].to(self.device)
            labels: Tensor = data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            average_loss += loss.item()

            if i % 100 == 0:
                # Only print every 100 batches so that printing doesn't
                # slow down the model
                print(
                    f'Batch #{i}/{data_count}, Loss: {loss.item()}           ', end='\r')

        print()
        return average_loss / data_count
