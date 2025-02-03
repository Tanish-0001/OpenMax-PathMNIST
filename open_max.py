import torch
import torch.nn as nn
import scipy.spatial.distance as spd

from compute_openmax import recalibrate_scores
from evt_fitting import weibull_tailfitting
import matplotlib.pyplot as plt
from torchvision import transforms
from medmnist import PathMNIST
import numpy as np

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.2])
])

train_dataset = PathMNIST(split='train', download=False, transform=data_transform)
val_dataset = PathMNIST(split='val', download=False, transform=data_transform)

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def compute_distances(mean_activation_vector, penultimate_activations):
    distances = []
    for activation in penultimate_activations:
        eu_dist = spd.euclidean(mean_activation_vector, activation)
        cos_dist = spd.cosine(mean_activation_vector, activation)
        distances.append(eu_dist / 200. + cos_dist)
    return distances


def get_activations(model, x_batch=None):
    return model(x_batch)


def get_correct(pred, labels):
    pred_labels = torch.argmax(pred, dim=1)
    labels = torch.squeeze(labels, dim=1)
    res = torch.eq(pred_labels, labels)
    return res


def group_data(x, y):
    ind = y.argsort()  # get indices to sort x and y based on labels
    grouped_x = x[ind]
    grouped_y = y[ind]

    dataset_x = []
    dataset_y = []
    mark = 0  # beginning of current group

    for a in range(len(grouped_y) - 1):
        if grouped_y[a] != grouped_y[a + 1]:
            dataset_x.append(grouped_x[mark:a + 1])
            dataset_y.append(grouped_y[mark:a + 1])
            mark = a + 1  # the next group will start at a + 1

        if a == len(grouped_y) - 2:
            dataset_x.append(grouped_x[mark:len(grouped_y)])
            dataset_y.append(grouped_y[mark:len(grouped_y)])

    return dataset_x, dataset_y


def get_mav(model, data):
    x_train, y_train, x_val, y_val = data
    x_all = torch.cat((x_train, x_val))
    y_all = torch.cat((y_train, y_val))
    pred = model(x_all)
    index = get_correct(pred, y_all)

    x1 = x_all[index].squeeze(dim=1)  # those examples that were correctly classified
    y1 = y_all[index].squeeze(dim=1)  # corresponding labels

    grp_x, grp_y = group_data(x1, y1)  # group them with respect to labels

    mean_activation_vectors = {}
    feature_distances = {}

    for u, v in zip(grp_x, grp_y):
        label = v[0]
        penultimate_activations = model(u)

        mean_activation_vector = torch.mean(penultimate_activations, dim=0)
        distances = compute_distances(mean_activation_vector.detach().numpy(), penultimate_activations.detach().numpy())

        mean_activation_vectors[label] = mean_activation_vector
        feature_distances[label] = distances

    return mean_activation_vectors, feature_distances


def get_weibull(mav, distances):
    weibull_model = {}
    for i in mav.keys():
        weibull = weibull_tailfitting(mav[i], distances[i], tailsize=25)
        weibull_model[labels[i]] = weibull

    return weibull_model


def compute_openmax(mav, distances, imagearr):

    weibull_model = get_weibull(mav, distances)
    openmax, softmax = recalibrate_scores(weibull_model, labels, imagearr)
    return openmax, softmax


if __name__ == '__main__':
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset) // 10, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset) // 10, shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_val, y_val = next(iter(val_loader))

    mav, distances = get_mav(model, (x_train, y_train, x_val, y_val))

    for i in range(10):
        sample = torch.randint(low=0, high=len(x_val), size=(1,)).item()
        test_x1 = x_val[sample].view(1, 3, 28, 28)
        test_y1 = y_val[sample].view(1, 1)

        activations = model(test_x1)
        openmax, softmax = compute_openmax(mav, distances, activations)

        print('Actual Label:', test_y1[0].item())
        print('Prediction Softmax:', torch.argmax(softmax))

        if torch.argmax(openmax) == 9:
            openmax = 'Unknown'
            print('Open max: ', openmax)
        else:
            print('Prediction openmax: ', torch.argmax(openmax))

