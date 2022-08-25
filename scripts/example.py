import torch
import torchvision
from LatentVariablesTour import LatentVariablesTour
from LinearMnistAutoencoder import AutoEncoder

if __name__ == "__main__":
    mnist_data = torchvision.datasets.MNIST(
        "./mnist",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = AutoEncoder(28 * 28)
    net.to(device)
    net.load_state_dict(torch.load("linear_mnist_autoenc.pth"))
    lvt = LatentVariablesTour(model=net, dataset=mnist_data, device=device)
    lvt.run()
