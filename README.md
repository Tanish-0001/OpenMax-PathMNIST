# OpenMax-PathMNIST
Implementation of the OpenMax layer in a deep network as described in the paper "Towards Open Set Deep Networks" presented at CVPR 2016 (https://arxiv.org/abs/1511.06233). Openmax is an alternative to the softmax layer used in ANNs for classification tasks. Openmax is designed for open datasets where we want the model to not misclassify unknown data into one of the labels it was trained on. Instead, we want it to simply return UNKNOWN in such cases.

Dataset used for the task is PathMNIST (Medical images from Colon Pathology) for training and DermaMNIST (images from a Dermatoscope) as the unknown dataset. Both datasets are from the MedMNIST collection of biomedical images.

First we train the model normally (train_model.ipynb). Then we find the Mean Activation Vectors (MAVs) for each label in the dataset. A mean activation vector is the mean of the penultimate activations of all correctly classified examples. At the same time we also store the distance of the MAV from the penultimate activations of each correctly classified example (Penultimate activation refers to the output of the last layer before applying softmax).

Once we have this data, we can fit a weibull distribution for each label. With this weibull distribution we can then modify the softmax scores as described in the paper to get the final openmax layer outputs.
