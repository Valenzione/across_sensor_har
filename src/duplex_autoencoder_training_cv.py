import os
import shutil

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (LSTM, Concatenate, Conv1D, Conv2DTranspose,
                          CuDNNLSTM, Dense, Dropout, Flatten, Input, Lambda,
                          RepeatVector, Reshape)
from keras.models import Model, Sequential
from tqdm import tqdm
from loguru import logger

from src.data.generators import create_generators
from src.data.shl_data import mean, shl_max, shl_min, std

x_min = shl_min()[np.newaxis, np.newaxis, :]
x_max = shl_max()[np.newaxis, np.newaxis, :]
x_mean = mean()
x_std = std()


def s2s_duplex_gen(generator, s1, s2):
    x_min = shl_min()[np.newaxis, np.newaxis, :]
    x_max = shl_max()[np.newaxis, np.newaxis, :]
    while True:
        x, y = next(generator)
        x = (x - x_min) / (x_max - x_min)
        x = x * 2 - 1
        s1_x = x[:,:,:,s1]
        s2_x = x[:,:,:,s2]
        yield s1_x, [s2_x, y]

def get_encoder_dense(inp):
    h = Flatten()(inp)
    h = Dense(1024, activation="relu")(h)
    h = Dense(512, activation="relu")(h)
    features = Dense(256, activation="relu", name="features")(h)
    h = Dropout(0.1)(features)
    return h

def get_decoder_dense(inp):
    h = Dense(512, activation="relu")(inp)
    h = Dense(1024, activation="relu")(h)
    h = Dense(1500, activation="tanh")(h)
    h = Reshape((500, 3))(h)
    return h

def get_duplex_classifier(inp):
    h = Dense(128, activation="relu")(inp)
    h = Dense(19, activation = "softmax")(h)
    return h

def get_callbacks(i, model_name):
    es = EarlyStopping(patience=5)
    mc = ModelCheckpoint(f"models/hips/best_fold{i}_{model_name}", save_best_only=True)
    rlr = ReduceLROnPlateau(patience=3)
    return [es, mc, rlr]

def get_model():
    inp = Input(batch_shape=(128, 500, 3))
    features = get_encoder_dense(inp)
    out = get_decoder_dense(features)
    out_cls = get_duplex_classifier(features)
    model = Model(inp, [out, out_cls])
    model.compile("rmsprop", ["mean_squared_error", "categorical_crossentropy"])
    return model
    
sensors = ["accel", "gyro", "mag"]
sources = [0]*3 + [1]*3 + [2]*3
destinations = [0, 1, 2]*3 
modalities = list(zip(sources, destinations))
model_names = [f"{sensors[x]}2{sensors[y]}_duplex" for (x, y) in modalities]


logger.add("duplex_ae_cv.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
logger.info("CV autoencoder fit process has started.")
for model_name, (in_sensor, out_sensor) in tqdm(list(zip(model_names, modalities))[8:], total=9, desc = "Modalities"):
    for i in tqdm(range(5), desc = "Folds", leave = False):
        if (model_name == "mag2mag_duplex") and (i < 2):
                continue
        else:
                train_fnames = np.load(f"data/filenames/s2s_fold{i}/train_filenames.npy")
                val_fnames = np.load(f"data/filenames/s2s_fold{i}/val_filenames.npy")
                train_generator, test_generator = create_generators("hips", f"s2s_fold{i}")
                train_gen, test_gen = s2s_duplex_gen(train_generator, in_sensor, out_sensor), s2s_duplex_gen(test_generator, in_sensor, out_sensor)
                model = get_model()
                logger.info(f"Processing {model_name}_fold_{i}...")
                model.fit_generator(train_gen, steps_per_epoch=748, epochs=300,
                        callbacks = get_callbacks(i, model_name), verbose = 0,
                        validation_data = test_gen, validation_steps = 187)
                logger.info(f"{model_name} finished")
