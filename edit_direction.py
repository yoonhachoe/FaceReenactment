import numpy as np
import torch
import os
import pickle
from dataset_svm import MyDatasetSVM
from torch.utils.data import DataLoader
from sklearn import svm

def train_boundary(mode):
    """Trains boundary in latent space with given emotion label.
    Given a collection of latent codes and the attribute labels for the
    corresponding images, this function will train a linear SVM by treating it as
    a bi-classification problem.
    NOTE: The returned boundary is with shape (1, latent_space_dim), and also
    normalized with unit norm.
    ---- Adapted from: https://github.com/genforce/interfacegan/blob/acec139909fb9aad41fbdbbbde651dfc0b7b3a17/utils/manipulator.py#L12

    Args:
        mode: which emotional attribute to take the latent codes from (.pkl-files)
    Returns:
        A decision boundary with type `numpy.ndarray`.
    """

    # Preparing and splitting data
    dataset = MyDatasetSVM(mode)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

    latents = []
    labels = []
    for i in range(len(train_data)):
        latent, label = train_data[i]
        latent = latent.cpu()
        latent = latent.detach().numpy()
        label = label.cpu().detach().numpy()
        latents.append(latent)
        labels.append(label)

    labels = np.ravel(labels)
    latent_space_dim = latents[0].size
    print("Latent space dim:", latent_space_dim)
    print("Size of dataset:", len(dataset))


    # Train the SVM Classifier
    clf = svm.SVC(kernel='linear')
    classifier = clf.fit(latents, labels)

    # Check the accuracy
    test_latents = []
    test_labels = []
    for i in range(len(test_data)):
        test_latent, test_label = test_data[i]
        test_latent = test_latent.cpu()
        test_latent = test_latent.detach().numpy()
        test_label = test_label.cpu().detach().numpy()
        test_latents.append(test_latent)
        test_labels.append(test_label)
    test_labels = np.ravel(test_labels)

    val_prediction = classifier.predict(test_latents)
    correct_num = np.sum(test_labels == val_prediction)
    print('Test Accuracy: ', (correct_num/len(test_latents)))

    # Get the boundary
    a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
    boundary = a / np.linalg.norm(a)
    return boundary

def linear_interpolate(boundary,
                       start_distance=-10.0,
                       end_distance=10.0,
                       steps=10):
    """Manipulates the given latent code with respect to a particular boundary.
    This function takes a boundary as inputs, and outputs a collection of manipulated latent codes.
    For example, let `steps` to be 10, then the input `latent_code` is with shape [1, latent_space_dim], input
    `boundary` is with shape [1, latent_space_dim] and unit norm, the output is
    with shape [10, latent_space_dim]. The first output latent code is
    `start_distance` away from the given `boundary`, while the last output latent
    code is `end_distance` away from the given `boundary`. Remaining latent codes
    are linearly interpolated.
    NOTE: Distance is sign sensitive.
    ---- Adapted from: https://github.com/genforce/interfacegan/blob/acec139909fb9aad41fbdbbbde651dfc0b7b3a17/utils/manipulator.py#L12

    Args:
        boundary: The semantic boundary as reference.
        start_distance: The distance to the boundary where the manipulation starts. (default: -10.0)
        end_distance: The distance to the boundary where the manipulation ends.(default: 10.0)
        steps: Number of steps to move the latent code from start position to end position. (default: 10)
    """
    # Getting the latent codes
    with open('latents_neutral.pkl', 'rb') as f:
        latent_codes = pickle.load(f)
    latent_code = latent_codes[119].reshape(1, -1) #randomly get the 120th sample of the given neutral faces to be manipulated
    latent_code = latent_code.cpu()
    latent_code = latent_code.detach().numpy()

    assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
            len(boundary.shape) == 2 and
            boundary.shape[1] == latent_code.shape[-1])

    linspace = np.linspace(start_distance, end_distance, steps)
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    edited_latents = latent_code + linspace * boundary
    return edited_latents


if __name__ == "__main__":
    device = torch.device('cpu')
    boundary = train_boundary('happy')

    #save the currently used boundary for later use
    with open('boundary.pkl', 'wb') as f:
        pickle.dump(boundary, f)
    edited_latents = linear_interpolate(boundary)
    edited_latents = torch.from_numpy(edited_latents)

    # create latents.pkl file feedable to visualization with StyleGAN generator
    edited_latents = edited_latents.reshape(-1, 1, 16, 512)
    latents_list = []
    for i in range(edited_latents.shape[0]):
        latents_list.append(edited_latents[i])

    #save edited latents as list -> visualize with visualize.py
    with open('latents_edited_happy.pkl', 'wb') as g:
        pickle.dump(latents_list, g)


