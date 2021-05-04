from models.vgg_v2 import CIFARModel
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


writer = SummaryWriter('logs/vgg_v2')
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                        download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                         shuffle=True, num_workers=2)

dataiter = iter(dataloader)
images, labels = dataiter.next()  # type: ignore

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

net = CIFARModel(trainset, testset).to('cuda').load('outputs/30.04-17:06.pt').to('cuda')
writer.add_graph(net, images.to('cuda'))
writer.close()
