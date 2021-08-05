from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Subset, SubsetRandomSampler
from torchvision.transforms import transforms
import torch
from DataAugement import RandAugment
from sklearn.model_selection import train_test_split
import numpy as np


def load_gen_data_dir(data_dir, kfold, resize=(32, 32), batch_size=128, augment=False, labeled_size=0.25):
    """
    Generic Data Generator - all the images should be in the data_dir devided to directories according to classes
    :param data_dir: path to directory with images
    :param kfold: size of k-fold
    :param resize: resize image size
    :param batch_size:
    :param augment: should augement images using RandAugment in DataAugement.py
    :param labeled_size: size ratio of the data that will be in the labled generator
    :return: It returns 2 generators with k-fold each, The generators both returns lables.
    The first one should be the unlabled data, the second one should be the labled data
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize)
    ])
    if augment:
        transform.transforms.insert(0, RandAugment(4, 10))

    image_datasets = datasets.ImageFolder(data_dir, transform=transform)
    targets = np.array(image_datasets.targets)
    indices_to_use = stratify_limit_samples(targets, 1000)
    targets = np.array(targets)[indices_to_use]

    unlabeled_idx, labeled_idx = train_test_split(np.arange(len(targets)), test_size=labeled_size,
                                                  shuffle=True, stratify=targets)

    unlabeled_set = Subset(image_datasets, unlabeled_idx)
    labeled_set = Subset(image_datasets, labeled_idx)

    class_names = image_datasets.classes

    return _k_fold_generator(unlabeled_set, kfold, int(batch_size * (1 - labeled_size))), \
           _k_fold_generator(labeled_set, kfold, int(batch_size * labeled_size)), class_names


def _k_fold_generator(dataset, kfold, batch_size):
    # Do K fold split
    dataset_size = len(dataset)
    fraction = 1 / kfold
    seg = int(dataset_size * fraction)

    for fold in range(kfold):
        trll = 0
        trlr = fold * seg
        vall = trlr
        valr = fold * seg + seg
        trrl = valr
        trrr = dataset_size

        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall, valr))

        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
        yield train_loader, val_loader


def stratify_limit_samples(targets, final_size):
    """
    :param targets: ndarray of the labels according
    :param final_size:
    :return: Indices list size of final_size startified according to original distribution of labels
    """
    uniques = np.unique(targets)
    classes_indicies = list()
    for idx, unique in enumerate(uniques):
        classes_indicies.append(np.argwhere(targets == unique).reshape(-1)[:final_size//len(uniques)])

    return np.array([item for sublist in classes_indicies for item in sublist])


if __name__ == '__main__':
    # unlabeled_gen, labeled_gen, classes = load_gen_data_dir("Datasets\\shapes", kfold=10)
    # for fold, ((unlabeled_train, _), (labeled_train, labeled_val)) in enumerate(zip(unlabeled_gen, labeled_gen)):
    #     print(fold)
    load_gen_data_dir("Datasets\\shapes", kfold=10)
