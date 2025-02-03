import numpy as np
from scipy.stats import weibull_min


def weibull_tailfitting(mav, distances, tailsize):

    """ Read through distance files, mean vector and fit weibull model
    for each category

    Input:
    --------------------------------
    meanfiles_path : contains path to files with pre-computed mean-activation
     vector
    distancefiles_path : contains path to files with pre-computed distances
     for images from MAV
    labellist : ImageNet 2012 labellist

    Output:
    --------------------------------
    weibull_model : Perform EVT based analysis using tails of distances and
    save weibull model parameters for re-adjusting softmax scores
    """
    distance = np.array(distances)

    # Sort distances and take the largest tailsize values
    tail_to_fit = np.sort(distance)[::-1]

    shape, loc, scale = weibull_min.fit(tail_to_fit, floc=0)

    weibull_model = {
        'mean_vec': mav,
        'weibull_params': {
            'shape': shape,
            'loc': loc,
            'scale': scale
        }
    }

    return weibull_model

