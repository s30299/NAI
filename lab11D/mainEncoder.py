import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)


test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64


train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


latent_dim = 64

class Autoencoder(nn.Module):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim

    self.encoder = nn.Sequential(
        nn.Conv2d(1, 16, (3,3), stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, (3,3), stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
    )

    self.decoder = nn.Sequential(
        nn.Linear(128, 32 * 7 * 7),
        nn.ReLU(),
        nn.Unflatten(1, (32, 7, 7)),
        nn.ConvTranspose2d(32, 16, (3,3), stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 1, (3,3), stride=2, padding=1, output_padding=1),
        nn.Sigmoid(),
    )

  def forward(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded



model = Autoencoder(latent_dim).to(device)
print(model)
loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X= X.to(device)


        pred = model(X)
        loss = loss_fn(pred, X)


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, X).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
torch.save(model.state_dict(), "autoencoder.pth")
print("Saved PyTorch Model State to autoencoder.pth")

