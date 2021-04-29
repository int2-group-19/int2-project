import torchvision
from torchvision import transforms
from models.model_dob9601 import CIFARModel


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                       download=True, transform=transform)

model = CIFARModel(trainset, testset).to('cuda')
model.run()
