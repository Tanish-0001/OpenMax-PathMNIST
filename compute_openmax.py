import scipy.spatial.distance as spd
from torch.nn.functional import softmax
from evt_fitting import query_weibull
import torch


def compute_distance(scores, mean):
    """
    Compute the euclidean-cosine distance between the scores and the mean vector

    Input:
    -----------------
    scores: the scores tensor
    mean: the mean vector

    Output:
    -----------------
    The euclidean-cosine distance
    """
    eu_dist = spd.euclidean(scores, mean)
    cos_dist = spd.cosine(scores, mean)
    return eu_dist / 200. + cos_dist


def computeOpenMaxProbability(openmax_penultimate, openmax_score_u):
    """
    Convert the scores in probability value using openmax

    Input:
    ---------------
    openmax_penultimate : modified penultimate layer from Weibull based computation
    openmax_score_u : degree

    Output:
    ---------------
    modified_scores : probability values modified using OpenMax framework,
    by incorporating degree of uncertainty/openness for a given class

    """
    prob_scores = torch.exp(openmax_penultimate)
    total_denominator = torch.sum(torch.exp(openmax_penultimate)) + torch.sum(torch.exp(openmax_score_u))
    prob_scores /= total_denominator
    prob_unknown = torch.exp(torch.sum(openmax_score_u)) / total_denominator

    prob_unknown = prob_unknown.view(1)
    modified_scores = torch.cat((prob_scores, prob_unknown), dim=0)
    return modified_scores


# ---------------------------------------------------------------------------------


def recalibrate_scores(weibull_model, labellist, penultimate_activations, alpharank=6):
    """
    Given activation features for an image and list of weibull models for each class,
    re-calibrate scores

    Input:
    ---------------
    weibull_model : pre-computed weibull_model obtained
    labellist : list containing category ids
    penultimate_activations : activations of layer before softmax for a particular image

    Output:
    ---------------
    openmax_probab: Probability values for a given class computed using OpenMax
    softmax_probab: Probability values for a given class computed using
     SoftMax (these were precomputed from caffe architecture. Function returns
     them for the sake of convienence)

    """

    scores = softmax(penultimate_activations, dim=0)  # get base scores after applying softmax
    penultimate_activations = penultimate_activations.squeeze()

    ranked_list = scores.argsort(descending=True).ravel()

    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    ranked_alpha = torch.zeros(10)

    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    openmax_penultimate, openmax_score_u = [], []

    for categoryid in range(len(labellist)):
        # get distance from mean vector
        category_weibull = query_weibull(labellist[categoryid], weibull_model)
        distance = compute_distance(penultimate_activations.cpu().detach().numpy(), category_weibull[0].cpu().detach().numpy())

        wscore = category_weibull[2][0].w_score(distance)  # category_weibull = [mean_vec, distances, weibull_model]

        # modify scores
        modified_penultimate_score = penultimate_activations[categoryid] * (1 - wscore * ranked_alpha[categoryid])
        openmax_penultimate += [modified_penultimate_score]
        openmax_score_u += [penultimate_activations[categoryid] - modified_penultimate_score]

    openmax_penultimate = torch.tensor(openmax_penultimate)
    openmax_score_u = torch.tensor(openmax_score_u)

    # normalize
    openmax_score_u = openmax_score_u / torch.sum(openmax_score_u)

    # get openmax probability
    openmax_probab = computeOpenMaxProbability(openmax_penultimate, openmax_score_u)

    softmax_probab = scores.ravel()

    return openmax_probab, softmax_probab
