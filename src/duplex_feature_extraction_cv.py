from keras.models import load_model, Model
import numpy as np
import os
from src.data.shl_data import shl_min, shl_max, mean, std
from tqdm import tqdm
import seaborn as sns
import shutil
from loguru import logger

x_min = shl_min()[np.newaxis, np.newaxis, :]
x_max = shl_max()[np.newaxis, np.newaxis, :]
x_mean = mean()
x_std = std()

base = "data/interim/hips/data"

logger.add("duplex_fe_cv.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
logger.info("CV feature extraction has started.")

sensors = ["accel", "gyro", "mag"]
sources = [0]*3 + [1]*3 + [2]*3
destinations = [0, 1, 2]*3 
modalities = list(zip(sources, destinations))
model_names = [f"{sensors[x]}2{sensors[y]}_duplex" for (x, y) in modalities]

for model_name, (in_sensor, out_sensor) in tqdm(list(zip(model_names, modalities))[5:], total=9, desc = "Modalities"):
    for i in tqdm(range(5), desc = "Folds", leave = False):

        os.makedirs(f"data/interim/hips/best_fold{i}_{model_name}_features/")
        
        model = load_model(f"models/hips/best_fold{i}_{model_name}")
        feature_encoder = Model(model.input, model.get_layer("features").output)
        rmses = []
        for fname in tqdm(os.listdir(base), desc = "files", leave = False):
            arr = np.load(base + "/" + fname)
            x = (arr - x_mean) / x_std
            x = (x - x_min) / (x_max - x_min)
            x = x * 2 - 1

            features = feature_encoder.predict(x[:,:,:, in_sensor])
            np.save(f"data/interim/hips/best_fold{i}_{model_name}_features/{fname}", features)
            rmse = np.mean(np.square(model.predict(x[:, :, :, in_sensor], verbose=0)[0] - x[:, :, :, out_sensor]), axis = 1)

            rmses.extend(rmse)
        logger.info(f"{model_name} fold {i} finished with rmse = {np.mean(rmses)}")