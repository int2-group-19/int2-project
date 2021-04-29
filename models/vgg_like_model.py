from __future__ import annotations
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.tensor import Tensor
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_transform = transforms.Compose(
    [  # transforms.RandomRotation(30),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        #  transforms.ColorJitter(brightness=0.4, contrast=0.4,
        #                         saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                        download=True, transform=train_transform)
print(trainset)
testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                       download=True, transform=transform)


class CIFARModel(nn.Module):

    epochs = 100
    batch_size = 64

    def __init__(self, dataset: Dataset, testset: Dataset):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Conv2d(64, 64, 5,padding=2),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(128, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

        )

        self.linear = nn.Sequential(
            nn.Linear(256 * 3 * 3, 120),
            nn.ReLU(),
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                      shuffle=True, num_workers=2)

        self.testset = testset
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=2)

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), momentum=0.9, lr=0.004)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        # print(x.shape)
        x = x.view(-1, 256 * 3 * 3)
        x = self.linear(x)

        return x

    def train(self) -> float:
        """
        Run a singular epoch on the given dataset.

        :param loader: The dataloader to load the data from.

        :returns: The average loss across this epoch.
        """
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

    def run(self):
        """
        Run the model on a given dataset.

        :param dataset: The dataset to run on
        """

        for epoch in range(self.epochs):
            print(f'Epoch #{epoch} ')

            average_loss = self.train()
            print(f'[DONE] [Average Loss: {average_loss}]')
            print(self.calculate_accuracy())

    def calculate_accuracy(self) -> float:
        """
        Calculate the accuracy of the model by testing it on its dataset.

        :returns: A float representing the accuracy of the model.
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(
                    self.device), data[1].to(self.device)
                outputs = self(images)

                _, predicted = torch.max(outputs.data, 1)

                # The number of items is the 0th dimension
                total += labels.size(0)

                # The sum of a boolean tensor is equal to the number of "Trues" in it
                correct += (predicted == labels).sum().item()

        return correct / total

    @classmethod
    def load(cls, path: str) -> CIFARModel:
        """
        Load the model after it has been serialised to a file.

        :param path: The path of the file.

        :returns: An instance of CIFARModel, constructed from the file.
        """
        return torch.load(path)

    def save(self, path: str) -> None:
        """
        Save the model to a file.

        :param path: The path to save to.
        """
        torch.save(self, path)


model = CIFARModel(trainset, testset).to('cuda')
model.run()
print(model.calculate_accuracy())
