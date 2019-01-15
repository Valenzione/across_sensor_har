import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.data.shl_data as shl
import os
dirname = os.path.dirname(__file__)


dirname = os.path.dirname(__file__)

def _split_filenames(filenames, splits = 6):
    random.shuffle(filenames)
    filenames = filenames[:np.floor(len(filenames)/splits)*splits] # To avoid uneven splits in np.split
    return np.split(filenames, splits, axis=0)


def create_generators(mode: str, exp_name: str, val_ratio = 0.1):
    """
    Create train generator and validation generaton for experiment name.
    If new, file with filenames will be created. Otherwise, such file will be used.
    """
    filenames_path = f"{dirname}/../../data/filenames"
    if not os.path.exists(f"{filenames_path}/{exp_name}"):
        filenames = os.listdir(f"{dirname}/../../data/processed/X/hips")
        np.random.shuffle(filenames)
        train_fnames, val_fnames = np.split(filenames, [int(len(filenames)*(1-val_ratio))])
        os.makedirs(f"{filenames_path}/{exp_name}/")
        np.save(f"{filenames_path}/{exp_name}/train_filenames", train_fnames)
        np.save(f"{filenames_path}/{exp_name}/val_filenames", val_fnames)
    else:
        train_fnames = np.load(f"{filenames_path}/{exp_name}/train_filenames.npy")
        val_fnames = np.load(f"{filenames_path}/{exp_name}/val_filenames.npy")
    return (filename_generator(mode, train_fnames), filename_generator(mode, val_fnames))

# TODO include different sensors?
def filename_generator(mode, filenames, infinite = True, normalize = True):
    """Returns generator from direcetory with /data/ and /labels/ subfolder and multiple .npy files in it

    Args:
        infinite: type of generator
        normalize: standard normalization

    Returns:
        generator
    """
    if normalize:
        X_mean = shl.mean()
        X_std  = shl.std()
    while True:
        random.shuffle(filenames)
        for file in filenames:
            X = np.load(f"{dirname}/../../data/interim/{mode}/data/{file}")
            if normalize:
                X = (X - X_mean)/X_std
            y = np.load(f"{dirname}/../../data/interim/{mode}/labels_sp/{file}")
            yield X, y
        if not infinite:
            break

def cv_generator(directory, folds=6, normalize=True):
    """Makes cross validation generators from data folder.

    Args:
        folds: number of folds
        normalize: standard normalization

    Returns:
        (train_fnames, val_fnames, train_generator, val_generator)
    """
    filenames = os.listdir(directory+"/data")
    for train, val in _split_filenames(filenames, splits=folds):
        yield train, val, filename_generator(directory, train, True, normalize), filename_generator(directory, val, False, normalize)
