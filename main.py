import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST, DermaMNIST
from open_max import get_mav, compute_openmax


class PathNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(PathNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=5),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Sequential(nn.Linear(in_features=64 * 8 * 8, out_features=512), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=512, out_features=128), nn.ReLU())
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool((self.conv4(x)))

        x = x.view(-1, 64 * 8 * 8)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = PathNet(3, 9)
model.load_state_dict(torch.load('trained_model.pth', weights_only=True))
model.eval()

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.2])
])

# train and validation datasets
train_dataset = PathMNIST(split='train', download=False, transform=data_transform)
val_dataset = PathMNIST(split='val', download=False, transform=data_transform)
test_dataset = PathMNIST(split='test', download=False, transform=data_transform)

# unknown dataset - DermaMNIST because it has the same input dimensions and channels as original dataset
unknown_dataset = DermaMNIST('test', download=True, transform=data_transform)

# Get mean activation vectors and distances of correctly predicted examples from training set
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=False)
mav, distances = get_mav(model, train_loader)

# load validation and unknown datasets to compare softmax and openmax
val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
unknown_dataloader = DataLoader(unknown_dataset, batch_size=1, shuffle=False)

x_val, y_val = next(iter(val_loader))
x_unknown, y_unknown = next(iter(unknown_dataloader))


def accuracy_known(model, test_loader, mav, distances):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0

        for images, labels in test_loader:
            labels = labels.view(-1)
            outputs = model(images)

            openmax, _ = compute_openmax(mav, distances, outputs)

            predicted = torch.argmax(openmax)
            total += labels.size(0)
            correct += torch.eq(predicted, labels).sum().item()

        acc = correct / total

    return acc


def accuracy_unknown(model, unknown_loader, mav, distances):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0

        for images, labels in unknown_loader:
            outputs = model(images)

            openmax, _ = compute_openmax(mav, distances, outputs)

            predicted = torch.argmax(openmax)
            actual = torch.full(predicted.size(), 9, dtype=torch.long, device=predicted.device)

            total += labels.size(0)
            correct += torch.eq(predicted, actual).sum().item()

        acc = correct / total

    return acc


print(f'Known Accuracy: {accuracy_known(model, test_loader, mav, distances) * 100:.2f}%')
print(f'Unknown Accuracy: {accuracy_unknown(model, unknown_dataloader, mav, distances) * 100:.2f}%')

# compare on data from same distribution as training data
for _ in range(10):
    sample = torch.randint(low=0, high=len(x_val), size=(1,)).item()
    test_x1 = x_val[sample].view(1, 3, 28, 28)
    test_y1 = y_val[sample]

    activations = model(test_x1)
    openmax, softmax = compute_openmax(mav, distances, activations)

    print('Actual Label:', test_y1.item())
    print('Prediction Softmax:', torch.argmax(softmax))

    if torch.argmax(openmax) == 9:  # originally labels are from 0 to 8. So a label of 9 indicates UNKNOWN
        openmax = 'UNKNOWN'
        print('Prediction openmax: UNKNOWN')
    else:
        print('Prediction openmax: ', torch.argmax(openmax))

    print("-----------------------------------")

# test on unknown data
print("\n===========Testing on unknown data============\n")

for _ in range(10):
    sample = torch.randint(low=0, high=len(x_unknown), size=(1,)).item()
    test_x1 = x_unknown[sample].view(1, 3, 28, 28)
    test_y1 = y_unknown[sample]

    output = model(test_x1)

    openmax, softmax = compute_openmax(mav, distances, output)

    print('Actual Label:', test_y1.item())
    print('Prediction Softmax:', torch.argmax(softmax))

    if torch.argmax(openmax) == 9:
        print('Prediction openmax: UNKNOWN')
    else:
        print('Prediction openmax: ', torch.argmax(openmax))
