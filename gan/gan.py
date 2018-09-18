import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False),
            torch.nn.ReLU(inplace=True),

            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)[:, :, 2:-2, 2:-2]

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.layers(x).view(-1, 1)

def transform(img):

    pass

def fit():
    dataset = torchvision.datasets.MNIST(root='./', download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1204, ), (0.2986, ))
    ]))

    num_epochs = 100
    batch_size = 8
    gen = Generator()
    gen_opt = torch.optim.Adam(gen.parameters(), lr=0.0002)
    disc = Discriminator()
    disc_opt = torch.optim.Adam(disc.parameters(), lr=0.0002)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in tqdm(range(num_epochs)):
        for data, labels in tqdm(loader):
            # 1.1. Training discriminator on on real images
            labels[:] = 1
            labels = labels.float()
            disc_opt.zero_grad()
            logits = disc(data)
            disc_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[:, None])
            disc_loss.backward()
            disc_opt.step()

            # 1.2. Training discriminator on fake images
            noise = torch.randn(batch_size, 100, 1, 1)
            fake_data = gen(noise).detach()
            fake_labels = torch.zeros(batch_size)
            disc_opt.zero_grad()
            logits = disc(fake_data)
            disc_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, fake_labels[:, None])
            disc_loss.backward()
            disc_opt.step()

            # 2. Training the generator
            gen_opt.zero_grad()
            noise = torch.randn(batch_size * 2, 100, 1, 1)
            fake_data = gen(noise)
            fake_labels = torch.ones(batch_size * 2)
            logits = disc(fake_data)
            gen_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, fake_labels[:, None])
            gen_loss.backward()
            gen_opt.step()
            tqdm.write(f'disc loss: {disc_loss.data[0]:.5f}, gen loss: {gen_loss.data[0]:.5f}')
            # img = fake_data[0].detach().numpy()
            # plt.clf()
            # plt.imshow(img[0, :, :])
            # plt.pause(1e-7)

if __name__ == '__main__':
    fit()
