from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.tensor import Tensor
import torch.utils.data
from torch.utils.data.dataset import Dataset


class CIFARModel(nn.Module):

    # Set to -1 to use automatic number of epochs
    epochs = -1

    batch_size = 64

    def __init__(self, dataset: Dataset, testset: Dataset):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 18, 5),  # Reduces size by 4px  (28px)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # Halves image size    (14px)
            nn.Conv2d(18, 64, 5),  # Reduces size by 4px  (10px)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # Halves image size    (5px)
        )

        self.linear = nn.Sequential(
            nn.Linear(64 * 5 * 5, 120),
            nn.Linear(120, 60),
            nn.Linear(60, 10)
        )

        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                      shuffle=True, num_workers=2)

        self.testset = testset
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=2)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), momentum=0.9, lr=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = x.view(-1, 64 * 5 * 5)  # -1 means infer this dimension
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
                print(f'Batch #{i}/{data_count}, Loss: {loss.item()}           ', end='\r')

        print()
        return average_loss / data_count


    def run(self):
        """
        Run the model on a given dataset.

        :param dataset: The dataset to run on
        """

        max_accuracy = 0.0

        if self.epochs == -1:
            # True indicates a decrease in score relative to the best model 
            running_decreases = [False, False, False, False, False]
            epoch = 1

            while any(v == False for v in running_decreases):
                print(f'Epoch #{epoch} ')

                average_loss = self.train()

                accuracy = self.calculate_accuracy()

                del running_decreases[0]
                running_decreases.append(accuracy < max_accuracy)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy

                print(f'[DONE] [Average Loss: {average_loss}] [Accuracy: {accuracy*100:.2f}%]')

                epoch += 1

            print('Terminating due to 5 successive scores below best value')


        for epoch in range(self.epochs):
            print(f'Epoch #{epoch} ')

            average_loss = self.train()
            print(f'[DONE] [Average Loss: {average_loss}] [Accuracy: {self.calculate_accuracy()*100}%]')

    def calculate_accuracy(self) -> float:
        """
        Calculate the accuracy of the model by testing it on its testing dataset.

        :returns: A float representing the accuracy of the model.
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
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
