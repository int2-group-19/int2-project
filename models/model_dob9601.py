from __future__ import annotations
from models.base_model import BaseModel

import torch.nn as nn
import torch.optim as optim
from torch.tensor import Tensor
from torch.utils.data.dataset import Dataset


class CIFARModel(BaseModel):

    # Set to -1 to use automatic number of epochs
    epochs = -1

    batch_size = 32

    def __init__(self, dataset: Dataset, testset: Dataset):
        super().__init__(dataset, testset)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),  # 32px
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),  # 32px
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # Halves image size    (16px)

            nn.Conv2d(128, 256, 5, padding=2),  # 16px
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),  # 16px
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # Halves image size    (8px)

            nn.Conv2d(512, 1024, 5, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, 5, padding=2),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
        )

        self.linear = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 10)
        )

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), momentum=0.9, lr=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = x.view(-1, 512 * 4 * 4)  # -1 means infer this dimension
        x = self.linear(x)
        return x

    def train(self) -> float:
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
                print(f'Batch #{i}/{data_count}, Loss: {loss.item()}           ', end='\r')

        print()
        return average_loss / data_count

