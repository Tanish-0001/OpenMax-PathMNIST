import numpy as np
import scipy as sp
import torch
from torch.nn.functional import softmax


def weibull_cdf(x, shape, scale):
    """
    Compute the Weibull cumulative distribution function (CDF).
    """
    return 1 - np.exp(-((x / scale) ** shape))


def compute_w_score(channel_distance, weibull_params):
    """
    Compute the w_score using the Weibull CDF.

    Input:
    ---------------
    channel_distance : Distance between the activation vector and the mean vector.
    weibull_params : Tuple containing (shape, scale, location) of the Weibull distribution.

    Output:
    ---------------
    w_score : Probability that the distance is in the tail of the distribution.
    """
    shape, scale, location = weibull_params['shape'], weibull_params['scale'], weibull_params['loc']
    # Adjust the distance by the location parameter
    adjusted_distance = channel_distance - location
    # Compute the CDF
    cdf = weibull_cdf(adjusted_distance, shape, scale)
    # The w_score is the probability of being in the tail
    w_score = 1 - cdf
    return w_score


def compute_distance(scores, mean, distance_type='eucos'):
    """
    Compute the distance between the scores and the mean vector.
    """
    if distance_type == 'eucos':
        eu_dist = sp.spatial.distance.euclidean(scores, mean)
        cos_dist = sp.spatial.distance.cosine(scores, mean)
        return eu_dist / 200. + cos_dist
    elif distance_type == 'euclidean':
        return sp.spatial.distance.euclidean(scores, mean)
    elif distance_type == 'cosine':
        return sp.spatial.distance.cosine(scores, mean)
    else:
        raise ValueError(f"Unsupported distance type: {distance_type}")


def compute_openmax_probability(openmax_fc8, openmax_score_u):
    """
    Convert the scores into probability values using OpenMax.
    """
    prob_scores = torch.exp(openmax_fc8)
    total_denominator = torch.sum(torch.exp(openmax_fc8)) + torch.sum(torch.exp(openmax_score_u))
    prob_scores /= total_denominator
    prob_unknown = torch.exp(torch.sum(openmax_score_u)) / total_denominator
    prob_unknown = prob_unknown.view(1)

    modified_scores = torch.cat((prob_scores, prob_unknown))
    return modified_scores


def recalibrate_scores(weibull_model, labellist, imgarr, alpharank=6):
    """
    Recalibrate scores using OpenMax.

    Input:
    ---------------
    weibull_model : Pre-computed Weibull models for each class.
    labellist : List of class labels.
    imgarr : Dictionary containing 'scores' and 'fc8' for the input image.
    alpharank : Rank for alpha weighting.
    distance_type : Type of distance to use ('eucos', 'euclidean', 'cosine').

    Output:
    ---------------
    openmax_probab : OpenMax probability scores.
    softmax_probab : SoftMax probability scores.
    """
    # Extract scores and fc8 activations
    scores = softmax(imgarr, dim=1)
    fc8 = imgarr

    fc8 = fc8.squeeze()

    # Rank the scores and compute alpha weights
    ranked_list = scores.argsort(descending=True).ravel()

    alpha_weights = [(alpharank + 1 - i) / float(alpharank) for i in range(1, alpharank + 1)]

    ranked_alpha = torch.zeros(len(labellist))

    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Recalibrate each fc8 score
    openmax_fc8 = []
    openmax_score_u = []
    for categoryid in range(len(labellist)):
        # Get the Weibull model for the current class
        category_weibull = weibull_model[labellist[categoryid]]

        # Compute the distance between the fc8 activations and the mean vector
        channel_distance = compute_distance(fc8.detach().numpy(), category_weibull['mean_vec'].detach().numpy())

        # Compute the Weibull score
        wscore = compute_w_score(channel_distance, category_weibull['weibull_params'])

        # Modify the fc8 score using the Weibull score and alpha weight
        modified_fc8_score = scores[0][categoryid] * (1 - wscore * ranked_alpha[categoryid])
        openmax_fc8.append(modified_fc8_score)
        openmax_score_u.append(scores[0][categoryid] - modified_fc8_score)

    openmax_fc8 = torch.tensor(openmax_fc8)
    openmax_score_u = torch.tensor(openmax_score_u)

    # openmax_score_u = openmax_score_u / torch.sum(openmax_score_u)
    openmax_probab = compute_openmax_probability(openmax_fc8, openmax_score_u)
    softmax_probab = scores[0]

    return openmax_probab, softmax_probab