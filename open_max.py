import torch
import scipy.spatial.distance as spd
from tqdm import tqdm
from compute_openmax import recalibrate_scores
from evt_fitting import weibull_tailfitting


labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def compute_distances(mean_activation_vector, penultimate_activations):
    """
    Finds the euclidean-cosine distance between mean activation vector and each penultimate activation vector

    Input:
    ---------------
    mean_activation_vector : mean activation vector for a particular label
    penultimate_activations : penultimate activation vectors (i.e, before applying softmax layer)

    Output:
    ---------------
    distances : the distances
    """
    distances = []
    for activation in penultimate_activations:
        eu_dist = spd.euclidean(mean_activation_vector, activation)
        cos_dist = spd.cosine(mean_activation_vector, activation)
        distances.append(eu_dist / 200. + cos_dist)
    return distances


def get_correct(pred, labels):
    """
    Compares model predictions against ground truth labels

    Input:
    ---------------
    pred : model predictions
    labels : ground truth labels

    Output:
    ---------------
    res : Tensor where 1 corresponds to correct prediction and 0 otherwise
    """
    pred_labels = torch.argmax(pred, dim=1)
    labels = torch.squeeze(labels, dim=1)
    res = torch.eq(pred_labels, labels)
    return res


def group_data(x, y):
    """
    Groups data according to labels. (i.e, groups the 0s, 1s, 2s...)

    Input:
    ---------------
    x : features
    y : labels

    Output:
    ---------------
    dataset_x : Grouped features
    dataset_y : Corresponding grouped labels
    """
    ind = y.argsort()  # get indices to sort x and y based on labels
    grouped_x = x[ind]
    grouped_y = y[ind]

    dataset_x = []
    dataset_y = []
    mark = 0  # beginning of current group

    for a in range(len(grouped_y) - 1):
        if grouped_y[a] != grouped_y[a + 1]:  # current group ends
            dataset_x.append(grouped_x[mark:a + 1])
            dataset_y.append(grouped_y[mark:a + 1])
            mark = a + 1  # the next group will start at a + 1

        if a == len(grouped_y) - 2:  # end of dataset
            dataset_x.append(grouped_x[mark:len(grouped_y)])
            dataset_y.append(grouped_y[mark:len(grouped_y)])

    return dataset_x, dataset_y


def get_mav(model, data_loader):
    """
    Compute mean activation vectors (MAVs) and distances for each label in batches.

    Input:
    ---------------
    model : Trained model
    data_loader : DataLoader for the training dataset

    Output:
    ---------------
    mean_activation_vectors : Dictionary of MAVs for each label.
    feature_distances : Dictionary of distances for each label.
    """
    model.eval()

    mean_activation_vectors = {}
    feature_distances = {}

    label_counts = {}  # count number of examples of each label we have encountered

    for batch_x, batch_y in tqdm(data_loader, desc="Processing batches"):
        with torch.no_grad():

            pred = model(batch_x)
            index = get_correct(pred, batch_y)

            # Filter correctly classified examples
            x1 = batch_x[index].squeeze(dim=1)
            y1 = batch_y[index].squeeze(dim=1)

            # Group data by labels
            grp_x, grp_y = group_data(x1, y1)

            for u, v in zip(grp_x, grp_y):
                label = v[0].item()
                penultimate_activations = model(u)

                # Compute mean activation vector
                mean_activation_vector = torch.mean(penultimate_activations, dim=0)

                # Compute distances
                distances = compute_distances(mean_activation_vector.cpu().detach().numpy(), penultimate_activations.cpu().detach().numpy())

                # Update MAV and distances for the label
                if label in mean_activation_vectors:
                    # Update MAV incrementally - using the MAV computed so far along with the current
                    prev_mean = mean_activation_vectors[label]
                    prev_count = label_counts[label]
                    new_count = prev_count + len(u)

                    mean_activation_vectors[label] = (prev_mean * prev_count + mean_activation_vector * len(u)) / new_count
                    label_counts[label] = new_count

                    feature_distances[label].extend(distances)

                else:
                    # Initialize
                    mean_activation_vectors[label] = mean_activation_vector
                    feature_distances[label] = distances
                    label_counts[label] = len(u)

    return mean_activation_vectors, feature_distances


def get_weibull(mav, distances):
    """
    Fits a weibull distribution over the mean activation vector and distances.

    Input:
    ---------------
    mav : Mean activation vectors for every label
    distances : Distances of the activations of correctly classified examples with the MAV

    Output:
    ---------------
    weibull_model : Dictionary of weibull distributions for each label.
    """
    weibull_model = {}

    # fit a weibull curve to every label
    for i in mav.keys():
        weibull = weibull_tailfitting(mav[i], distances[i], tailsize=25)
        weibull_model[labels[i]] = weibull

    return weibull_model


def compute_openmax(mav, distances, activations):
    """
    Apply the openmax layer on activations.

    Input:
    ---------------
    mav : The mean activation vectors
    distances : Distances of the activations of correctly classified examples with the MAV

    Output:
    ---------------
    openmax : output after applying the openmax layer
    softmax : output after applying the softmax layer
    """
    weibull_model = get_weibull(mav, distances)
    openmax, softmax = recalibrate_scores(weibull_model, labels, activations)
    return openmax, softmax
