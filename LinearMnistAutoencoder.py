import torch
import torch.nn as nn


class LeakyMish(nn.Module):
    def __init__(self, slope=0.4, balance=5):
        super().__init__()
        self.slope = slope
        self.balance = balance
        self.mish = nn.Mish()
        self.elu = nn.LeakyReLU(slope * (-1) ** (balance > 1))

    def forward(self, x):
        return self.balance * self.mish(x) + (1 - self.balance) * self.elu(x)


class Encoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 64)
        self.fc3 = torch.nn.Linear(64, 16)
        self.fc4 = torch.nn.Linear(16, 3)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 64)
        self.fc3 = torch.nn.Linear(64, 512)
        self.fc4 = torch.nn.Linear(512, output_size)
        self.activ = LeakyMish(0.8, 5)

    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        x = self.activ(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = nn.Unflatten(1, (1, 28, 28))(x)
        return x


class AutoEncoder(torch.nn.Module):
    def __init__(self, org_size):
        super().__init__()
        self.enc = Encoder(org_size)
        self.dec = Decoder(org_size)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x