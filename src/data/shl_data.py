import pandas as pd
import numpy as np
import os

dirname = os.path.dirname(__file__)

def mean():
    filename = os.path.join(dirname, "../../data/processed/mean.csv")
    mean_df = pd.read_csv(filename, header=None, index_col=0)
    return mean_df.iloc[:,0][np.r_[0:9, 10:19]].values.reshape((1, 1, 3, 6))
    
def std():
    filename = os.path.join(dirname, "../../data/processed/std.csv")
    mean_df = pd.read_csv(filename, header=None, index_col=0)
    return mean_df.iloc[:,0][np.r_[0:9, 10:19]].values.reshape((1, 1, 3, 6))

def shl_max():
    filename = os.path.join(dirname, "../../data/processed/max.npy")

    return np.load(filename)

def shl_min():
    filename = os.path.join(dirname, "../../data/processed/min.npy")
    return np.load(filename)

labels = [x.split("=")[0] for x in """Null= 0
    Still;Stand;Outside= 1
    Still;Stand;Inside= 2
    Still;Sit;Outside= 3
    Still;Sit;Inside= 4
    Walking;Outside= 5
    Walking;Inside= 6
    Run= 7
    Bike= 8
    Car;Driver= 9
    Car;Passenger= 10
    Bus;Stand= 11
    Bus;Sit= 12
    Bus;Up;Stand= 13
    Bus;Up;Sit= 14
    Train;Stand= 15
    Train;Sit= 16
    Subway;Stand= 17
    Subway;Sit= 18""".split("\n")]

feature_columns = """acc_x acc_y acc_z gyr_x gyr_y gyr_z mag_x mag_y mag_z orient_w orient_x orient_y orient_z grav_x grav_y grav_z lin_acc_x lin_acc_y lin_acc_z pressure ignore ignore""".split(" ")

body_locations = ["Bag", "Torso", "Hips", "Hand"]

users = ["User1", "User2", "User3"]

all_labels = {}
all_labels['Coarse'] = "Null Still Walking Run Bike Car Bus Train Subway".split()
all_labels['Fine'] = """Null Still;Stand;Outside Still;Stand;Inside Still;Sit;Outside Still;Sit;Inside Walking;Outside Walking;Inside Run Bike Car;Driver Car;Passenger Bus;Stand Bus;Sit Bus;Up;Stand Bus;Up;Sit Train;Stand Train;Sit Subway;Stand Subway;Sit""".split(" ")
all_labels['Road'] = "Null, City=1, Motorway=2, Countryside=3, Dirt road=4".split(",")
all_labels['Food'] = "Null, Eating=1, Drinking=2, Both=3".split(",")
all_labels['Tunnels'] = "Null Tunnel".split(" ")
all_labels['Social'] = "Null Social".split(" ")
all_labels['Traffic'] = "Null Heavy".split(" ")
