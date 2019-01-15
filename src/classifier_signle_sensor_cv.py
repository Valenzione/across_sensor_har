import os
import shutil

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (LSTM, Concatenate, Conv1D, Conv2DTranspose,
                          CuDNNLSTM, Dense, Dropout, Flatten, Input, Lambda,
                          RepeatVector, Reshape)
from keras.models import Model, Sequential, load_model
from loguru import logger
# from tqdm import tqdm

from src.data.generators import create_generators
from src.data.shl_data import mean, shl_max, shl_min, std


def s_gen(generator, s1):
    x_min = shl_min()[np.newaxis, np.newaxis, :]
    x_max = shl_max()[np.newaxis, np.newaxis, :]
    while True:
        x, y = next(generator)
        x = (x - x_min) / (x_max - x_min)
        x = x * 2 - 1
        s1_x = x[:, :, :, s1]
        yield s1_x,  y

def get_encoder_dense(inp):
    h = Flatten()(inp)
    h = Dense(1024, activation="relu")(h)
    h = Dense(512, activation="relu")(h)
    features = Dense(256, activation="relu", name="features")(h)
    h = Dropout(0.1)(features)
    return h

def get_classifier(inp):
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
    out_cls = get_classifier(features)
    model = Model(inp, out_cls)
    model.compile("rmsprop",  "categorical_crossentropy", metrics=["accuracy"])
    return model

x_min = shl_min()[np.newaxis, np.newaxis, :]
x_max = shl_max()[np.newaxis, np.newaxis, :]
x_mean = mean()
x_std = std()

base = "data/interim/hips/data"

logger.add("classifier_1_k_cv.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
logger.info("CV base classifier have started.")

sensors = ["accel", "gyro", "mag"]
sources = [0, 1, 2]


for in_sensor in sources:
    model_name = f"base_{sensors[in_sensor]}"
    for i in range(5):
        train_fnames = np.load(f"data/filenames/s2s_fold{i}/train_filenames.npy")
        val_fnames = np.load(f"data/filenames/s2s_fold{i}/val_filenames.npy")
        train_generator, test_generator = create_generators("hips", f"s2s_fold{i}")
        train_gen, test_gen = s_gen(train_generator, in_sensor), s_gen(test_generator, in_sensor)
        model = get_model()
        logger.info(f"Processing {model_name}_fold_{i}...")
        history = model.fit_generator(train_gen, steps_per_epoch=748, epochs=300,
                callbacks = get_callbacks(i, model_name), verbose = 1,
                validation_data = test_gen, validation_steps = 187)

        history_data =  np.array([history.history['val_acc'],
                                    history.history['val_loss'],
                                    history.history['acc'],
                                    history.history['loss']])

        np.save(f"histories/{model_name}_fold_{i}", history_data)  

        logger.info(f"{model_name} finished")
