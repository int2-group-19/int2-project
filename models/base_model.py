import abc
import datetime
from typing import Type, TypeVar

import matplotlib.pyplot as plt
from torch import nn
import torch
import torch.utils.data
from torch.utils.data.dataset import Dataset

V = TypeVar('V', bound='BaseModel')


class BaseModelMeta(abc.ABCMeta, type(nn.Module)):
    """Metaclass that combines nn.Module with ABCMeta."""
    ...


class BaseModel(nn.Module, abc.ABC, metaclass=BaseModelMeta):

    @property
    def batch_size(self) -> int:
        raise NotImplementedError()

    @property
    def epochs(self) -> int:
        raise NotImplementedError()

    def __init__(self, dataset: Dataset, testset: Dataset):
        super().__init__()

        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                      shuffle=True, num_workers=2)

        self.testset = testset
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=2)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def run(self):
        """Run the model on a given dataset."""

        if self.epochs == -1:
            self.__auto_run()
        else:
            self.__run()

    def __run(self):
        """
        Run the model for a given number of epochs.

        This function is called when epochs is set to any number other than -1.
        """
        accuracies = []

        for epoch in range(self.epochs):
            print(f'Epoch #{epoch} ')

            average_loss = self.train_model()
            accuracy = self.calculate_accuracy()
            accuracies.append(accuracy)

            print(f'[DONE] [Average Loss: {average_loss}] [Accuracy: {accuracy*100}%]')

        plt.plot(accuracies)
        plt.show()

    def __auto_run(self):
        """
        Run the model for an automatically determined number of epochs.

        The number of epochs is determined via running the model until 5 successive epochs
          don't beat the previous best score.
        """
        output_filename = datetime.datetime.now().strftime('%d.%m-%H:%M') + '.pt'
        print('Entering autorun mode')
        # True indicates a decrease in score relative to the best model 
        max_accuracy = 0.0
        running_decreases = [False for _ in range(30)]
        accuracies = []
        epoch = 1

        while any(v == False for v in running_decreases):
            print(f'Epoch #{epoch} ')

            average_loss = self.train_model()

            accuracy = self.calculate_accuracy()
            accuracies.append(accuracy)

            del running_decreases[0]
            if accuracy > max_accuracy:
                print('New best model - serialising')
                self.save(f'outputs/{output_filename}')
                max_accuracy = accuracy
                running_decreases.append(False)
            else:
                running_decreases.append(True)

            print(f'[DONE] [Average Loss: {average_loss}] [Accuracy: {accuracy*100:.2f}%]')

            epoch += 1

        plt.plot(accuracies)
        plt.show()

        print('Terminating due to 5 successive scores below best value')

    def calculate_accuracy(self) -> float:
        """
        Calculate the accuracy of the model by testing it on its testing dataset.

        :returns: A float representing the accuracy of the model.
        """
        correct = 0
        total = 0

        self.eval()
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self(images)

                _, predicted = torch.max(outputs.data, 1)

                # The number of items is the 0th dimension
                total += labels.size(0)

                # The sum of a boolean tensor is equal to the number of "Trues" in it
                correct += (predicted == labels).sum().item()

        self.train()
        return correct / total

    @abc.abstractmethod
    def train_model(self) -> float:
        """
        Run a singular epoch on the given dataset.

        :param loader: The dataloader to load the data from.

        :returns: The average loss across this epoch.
        """
        ...

    @classmethod
    def load(cls: Type[V], path: str) -> V:
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
