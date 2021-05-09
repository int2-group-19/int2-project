from __future__ import annotations

from models.base_model import BaseModel

import torch.nn as nn
import torch.optim as optim
from torch.tensor import Tensor
from torch.utils.data.dataset import Dataset


class CIFARModel(BaseModel):

    epochs = 100
    batch_size = 64

    def __init__(self, dataset: Dataset, testset: Dataset):
        super().__init__(dataset, testset)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.conv128 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv128to256 = nn.Conv2d(128, 256, 3, padding=1)

        self.conv256 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv256to512 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.MaxPool2d(3, 2, padding=1)
        )

        self.conv512 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(512 * 4 * 4, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 10),
            nn.Softmax(dim=0)
        )

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), momentum=0.9, lr=0.004)
        # self.optimizer = optim.Adagrad(self.parameters(), lr=4e-3)

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = self.conv1(x)

        x = self.conv128(y)
        x += y

        y = self.conv128to256(x)

        x = self.conv256(y)
        x += y

        y = self.conv256to512(x)

        x = self.conv512(y)
        x += y

        x = x.view(-1, 512 * 4 * 4)

        return self.linear(x)

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
