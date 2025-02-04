import libmr


def weibull_tailfitting(mav, distances, tailsize=25):
    """
    Read through distance files, mean vector and fits weibull model for each category

    Input:
    --------------------------------
    mav : the mean activation vector for a particular label
    distances : corresponding distances for that label
    tailsize : the tailsize to which weibull model is fit

    Output:
    --------------------------------
    weibull_model : Perform EVT based analysis using tails of distances and save weibull model
    parameters for re-adjusting softmax scores
    """

    weibull_model = {'distances': distances, 'mean_vec': mav, 'weibull_model': []}

    mr = libmr.MR()

    tailtofit = sorted(distances)[-tailsize:]  # take the biggest n distances
    mr.fit_high(tailtofit, len(tailtofit))
    weibull_model['weibull_model'] = [mr]

    return weibull_model


def query_weibull(category_name, weibull_model):
    """
    Query through dictionary for Weibull model.

    Input:
    ------------------------------
    category_name : name/id of category to query
    weibull_model: dictionary of weibull models

    Output:
    ------------------------------
    category_weibull : [mean_vec, distances, weibull_model]
    """

    category_weibull = [weibull_model[category_name]['mean_vec'],
                        weibull_model[category_name]['distances'],
                        weibull_model[category_name]['weibull_model']]

    return category_weibull
