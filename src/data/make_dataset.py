import os

import numpy as np
import pandas as pd
from keras.utils import to_categorical

from src.data.shl_data import body_locations, feature_columns, labels, users

dirname = os.path.dirname(__file__)

def generate_dataframe(dir_path, mode="Hips"):
    """Generate csv from {mode}.txt with dir_path pointing to directory with file"""
    data = pd.DataFrame(np.loadtxt(f"{dir_path}/{mode}_Motion.txt"), columns=["Time(ms)"] +feature_columns)
    label_data = pd.DataFrame(np.loadtxt(f"{dir_path}/Label.txt"), columns=["Time(ms)"] + labels)
    assert(label_data.shape[0] == data.shape[0])
    data.set_index(pd.to_datetime(data['Time(ms)'], unit='ms'), inplace=True)
    label_data.set_index(pd.to_datetime(label_data['Time(ms)'], unit='ms'), inplace=True)
    data = pd.concat([data, label_data], join='inner', axis=1)
    data.drop(["Time(ms)", "ignore"], axis=1, inplace = True)
    data.dropna(inplace=True)
    del label_data
    return data


def raw_to_hdf():
    for mode in body_locations:
        for user in users:
            _, subdirectories, _ = next(os.walk(f"../data/raw/{user}/"))
            for subdirectory in subdirectories:
                print("Loading...", user, subdirectory)
                data = generate_dataframe(f"{user}/{subdirectory}", mode=mode)
                data.to_hdf(f"../data/interim/{mode.lower()}_data/{user}_{subdirectory}.hdf5", key="shl")

# TODO: pass subsampling parameter to go from 100Hz to another value
def data_sequencer(data: pd.DataFrame, sequence_length = 500,
                         step = 250, batch_size = 128):
    """ Return generator producing sequences of fixed length from pd.DataFrame with some overlap
     in batches.
     """
    data.fillna(data.mean(), inplace = True)
    X = (data.iloc[:, np.r_[0:9, 10:19]]) # Select all columns except orient_w
    X = np.expand_dims(X, axis=2).reshape((X.shape[0], 3, 6)) # Stack sensors depthwise
    assert np.sum(X != X) == 0, f"Not cool - {np.sum(X != X)}, {X.shape}"

    Y = to_categorical(data['Fine'], num_classes=19)
    assert (X.shape[0] == Y.shape[0])
    sequence_starts = list(range(0, X.shape[0] - sequence_length, step))
    np.random.shuffle(sequence_starts)  # Shuffle sequences
    
    buffer_x = []
    buffer_y =[]

    for index in sequence_starts:
        x = np.expand_dims(X[index:index+sequence_length], axis = 0) 
        y = np.expand_dims(Y[index+sequence_length], axis = 0)

        buffer_x.append(x)
        buffer_y.append(y)
        
        if len(buffer_x) == batch_size:
            yield np.concatenate(buffer_x), np.concatenate(buffer_y)
            buffer_x.clear()
            buffer_y.clear()

def csv_to_npy_batches():
    for mode in ['hips']:
        for file in os.listdir(f"{dirname}/../../data/interim/{mode.lower()}_data/"):
            print(f"Uploading {file}!")
            data = pd.read_hdf(f"{dirname}/../../data/interim/{mode.lower()}_data/" + file, index_col="Time(ms)")
            for i, (X, Y) in enumerate(data_sequencer(data)):
                assert np.sum(X != X) == 0
                os.makedirs(f"{dirname}/../../data/processed/X/{mode.lower()}/", exist_ok=True)
                os.makedirs(f"{dirname}/../../data/processed/Y/{mode.lower()}/", exist_ok=True)
                np.save(f"{dirname}/../../data/processed/X/{mode.lower()}/{file}_{i}", X)
                np.save(f"{dirname}/../../data/processed/Y/{mode.lower()}/{file}_{i}", Y)


csv_to_npy_batches()