import random
from collections import Counter

import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from ignite.metrics import ConfusionMatrix
from ignite.engine import Engine
import seaborn as sns
import pandas as pd

transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)
batch_size = 64
test_loader = DataLoader(test_data, batch_size=batch_size)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28*10, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 5, (3,3), padding=1),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(5, 10, (3, 3), padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.cnn2(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model3.pth", map_location=device, weights_only=True))
model.eval()


def evaluate_step(engine, batch):
    x, y = batch
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        y_pred = model(x)
    return y_pred, y


evaluator = Engine(evaluate_step)

cm = ConfusionMatrix(num_classes=10)
cm.attach(evaluator, "cm")


evaluator.run(test_loader)

confusion_matrix = evaluator.state.metrics["cm"]

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

cm_df = pd.DataFrame(confusion_matrix.numpy(), index=classes, columns=classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

correct = confusion_matrix.trace()
total = confusion_matrix.sum()
accuracy = correct/total
print(f"Accuracy: {accuracy:.4f}")

y_true = []
y_pred = []

for batch in test_loader:
    x, y = batch
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    y_true.extend(y.cpu().numpy())
    y_pred.extend(pred.cpu().numpy())

precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

test_class_counts = Counter(test_data.targets.numpy())

plt.figure(figsize=(7, 6))

sns.barplot(x=list(test_class_counts.keys()), y=list(test_class_counts.values()))
plt.title("Super wykres Test data")
plt.xlabel("Klasa")
plt.ylabel("Ilość obrazów")

plt.tight_layout()
plt.show()

train_class_counts = Counter(training_data.targets.numpy())

plt.figure(figsize=(7, 6))

sns.barplot(x=list(train_class_counts.keys()), y=list(train_class_counts.values()))
plt.title("Super wykres Train data")
plt.xlabel("Klasa")
plt.ylabel("Ilość obrazów")

plt.tight_layout()
plt.show()
