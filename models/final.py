from __future__ import annotations
from models.base_model import BaseModel

import torch.nn as nn
import torch.optim as optim
from torch.tensor import Tensor
from torch.utils.data.dataset import Dataset

class CIFARModel(BaseModel):
 
    epochs = 400
    batch_size = 64
 
    def __init__(self, dataset: Dataset, testset: Dataset):
        super().__init__(dataset, testset)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()(1/5.5),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(1/5.5),
            nn.Conv2d(64, 64, 5,padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(1/5.5),
            nn.Dropout(0.1),
            nn.MaxPool2d(3, 2, padding=1),
 
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(1/5.5),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(1/5.5),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(1/5.5),
            nn.Dropout(0.1),
            nn.MaxPool2d(3, 2, padding=1),
 
 
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(1/5.5),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(1/5.5),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(1/5.5),
            nn.Dropout(0.1),
            nn.MaxPool2d(3, 2, padding=1),
 
 
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(1/5.5),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(1/5.5),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.MaxPool2d(3, 2, padding=1),
 
        )
 
        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2 * 2, 600),
            nn.LeakyReLU(1/5.5),
            nn.Dropout(0.5),
            nn.Linear(600, 600),
            nn.LeakyReLU(1/5.5),
            nn.Linear(600, 10),
            nn.Softmax(dim=1)
        )
 
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), momentum=0.9, lr=0.001)
 
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = x.view(-1, 512 * 2 * 2)
 
        x = self.linear(x)
 
        return x
 
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
