import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Model, Sequential, load_model
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.shl_data import labels, mean, shl_max, shl_min, std


def get_feature_xy(fnames, bases):
    label_base = "data/interim/hips/labels_sp/"
    features = []
    f_labels = []
    for fname in fnames:
        fs = []
        for base in bases:
            fs.append(np.load(base + "/" + fname))
        f = np.concatenate(fs, axis = 1)
        y = np.load(label_base + "/" + fname)
        features.extend(f)
        f_labels.extend(y)

    features = np.array(features)
    f_labels = np.array(f_labels)
    return features, f_labels

def get_model(inp_dim):
    model = Sequential()
    model.add(Dense(128, activation = "relu", input_shape=(inp_dim,)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(19, activation="softmax"))
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model

sensors = ["accel", "gyro", "mag"]
sources = [0]*3 + [1]*3 + [2]*3
destinations = [0, 1, 2]*3 
modalities = list(zip(sources, destinations))
model_names = [f"{sensors[x]}2{sensors[y]}_duplex" for (x, y) in modalities]

logger.add("file.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
logger.info("Cross-validation have started.")

modalities_combinations = [[((0, 0),), ((0, 0), (0, 1)), ((0, 0), (0, 2)), ((0, 0), (0, 1), (0, 2))],
                          [((1, 1),), ((1, 1), (1, 0)), ((1, 1), (1, 2)), ((1, 1), (1, 0), (1, 2))],
                          [((2, 2),), ((2, 2), (2, 0)), ((2, 2), (2, 1)), ((2, 2), (2, 0), (2, 1))]]
histories = {}
for modality in tqdm(modalities_combinations, desc="Modalities"):
    for sensors_combinations in tqdm(modality, desc="Feature combinations"):
        feature_combinations = [f"{sensors[sensor_c[0]]}2{sensors[sensor_c[1]]}_duplex" for sensor_c in sensors_combinations]
        model_name = "+".join(feature_combinations)
        logger.info(f"Processing combinations {model_name}")
        for i in tqdm(range(5), leave=False, desc="Folds"):
                logger.info(f"Cross-validating {model_name} fold_{i}.")

                train_fnames = np.load(f"data/filenames/s2s_fold{i}/train_filenames.npy")
                val_fnames = np.load(f"data/filenames/s2s_fold{i}/val_filenames.npy")

                feature_types = [f"data/interim/hips/best_fold{i}_{x}_features/" for x in feature_combinations]
                X_train, y_train = get_feature_xy(train_fnames, feature_types)
                X_val, y_val = get_feature_xy(val_fnames, feature_types)
            
                model = get_model(X_train.shape[1])
                history = model.fit(X_train, y_train, batch_size=512, epochs=300,
                                        validation_data=(X_val, y_val), verbose = 0)
                model.save(f"models/classifier_models/{model_name}")
                history_data =  np.array([history.history['val_acc'],
                                    history.history['val_loss'],
                                    history.history['acc'],
                                    history.history['loss']])
                np.save(f"histories/{model_name}_fold_{i}", history_data)  
                histories[f"{model_name}_{i}"] = history_data
                logger.info(f"{model_name} fold_{i} finished.")


logger.info("Finished cross-validation. Saving history .csv ...")
df = pd.DataFrame.from_dict(histories)
df.to_csv("histories_duplex.csv")
