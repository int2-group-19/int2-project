import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data.dataset import Dataset

from models.base_model import BaseModel

class CIFARModel(BaseModel):

    epochs = -1
    batch_size = 64

    def __init__(self, dataset: Dataset, testset: Dataset):

        super().__init__(dataset, testset)

        self.conv1 = nn.Sequential(

            nn.Conv2d(3, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.conv2  =  nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2)

        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(576, 16, 3, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(32 * 17 * 17, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.ReLU(),
            nn.LogSoftmax(dim=-1)
        )

        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.parameters(), momentum=0.9, lr=0.004)


    def forward(self, x: Tensor) -> Tensor:

        out1 = self.conv1(x)

        out2 = self.conv2(out1)

        agg1 = torch.cat([out1, out2], 1)

        out3 = self.conv3(out2)

        agg2 = torch.cat([agg1, out3], 1)

        out = self.conv4(agg2)

        out = out.view(-1, 32 * 17 * 17)

        out = self.linear(out)

        return out

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
