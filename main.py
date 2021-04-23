import torch
import torchvision

# Download the training and testing datasets
training_dataset = torchvision.datasets.CIFAR10(root='datasets', train=True, download=True)
testing_dataset = torchvision.datasets.CIFAR10(root='datasets', train=False, download=True)
