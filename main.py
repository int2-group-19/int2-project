import torchvision
from torchvision import transforms
from model2 import CIFARModel


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                        download=True, transform=transform)
model = CIFARModel(trainset).to('cuda')
model.run()
print(model.calculate_accuracy())
