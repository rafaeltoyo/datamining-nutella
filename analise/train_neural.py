# -*- coding: utf-8 -*-

# ==============================================================================
# Comentários, etc, etc...
# ==============================================================================

import pandas as pd
import numpy as np
import tflearn

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression

# ==============================================================================

DATASET_FILENAME = "monolith.csv"
TEST_FILENAME = "monolith_test.csv"

ATTRIBUTES = [
    "acc_precipitation",
    "age", # Se tirar isso aqui fica uma porcaria
    "temperature",
    #"windspeed",
    #"dewpoint",
    #"Soilwater_L1",
    #"Soilwater_L2",
    #"Soilwater_L3",
    #"Soilwater_L4",
    "month",
    #"year",
    #"BLDFIE_sl7", # Se colocar esse aqui fica uma porcaria
    #"CRFVOL_sl4",
    #"ORCDRC_sl6", # Erhm
    #"BDRLOG_BDRLOG_M", # Erhm
    #"CECSOL_sl4",
    #"CLYPPT_sl6",
]

MULTIPLY_FACTOR = [
    0.7,  # Field 0
    0.9,  # Field 1
    1.05, # Field 2
    0.9,  # Field 3
    1.1,  # Field 4
    1.0,  # Field 5
    1.0,  # Field 6
]
    
# ==============================================================================

def main():
    train_dataset = pd.read_csv(DATASET_FILENAME)
    test_dataset = pd.read_csv(TEST_FILENAME)
    
    full_dataset = pd.concat([train_dataset, test_dataset])
    for attribute in ATTRIBUTES:
        max_value = full_dataset[attribute].max()
        min_value = full_dataset[attribute].min()
        train_dataset[attribute] = (train_dataset[attribute] - min_value) / (max_value - min_value)
        test_dataset[attribute] = (test_dataset[attribute] - min_value) / (max_value - min_value)

    x_data_tr = train_dataset[ATTRIBUTES]
    y_data_tr = train_dataset["production"]
    
    x_data_te = test_dataset[ATTRIBUTES]
    id_data_te = test_dataset["Id"]
    
    #min_max_scaler = preprocessing.MinMaxScaler()
    #np_scaled = min_max_scaler.fit_transform(x_data_tr)
    #x_data_tr = pd.DataFrame(np_scaled)
    

    # SPLIT DATASET
    x_train, x_test, y_train, y_test = train_test_split(
            x_data_tr, y_data_tr, test_size=0.2)
    
    # FULL DATASET
    #x_train = x_data_tr
    #x_test = x_data_tr
    #y_train = y_data_tr
    #y_test = y_data_tr
    
    network = input_data(shape=[None, len(ATTRIBUTES)], name="Input_layer")
    network = fully_connected(network, 20, activation="relu", name="Hidden_layer_1")
    network = fully_connected(network, 20, activation="relu", name="Hidden_layer_2")
    network = fully_connected(network, 10, activation="relu", name="Hidden_layer_3")
    network = fully_connected(network, 1, activation="linear", name="Output_layer")
    network = regression(network, batch_size=64, optimizer='adam', learning_rate=0.001, loss="mean_square", metric="R2")
    
    model = tflearn.DNN(network)
    
    x_sarue = x_train.values
    y_sarue = y_train.values.reshape(-1, 1)
    x_weeb = x_test.values
    y_weeb = y_test.values.reshape(-1, 1)
    
    model.fit(x_sarue, y_sarue, show_metric=True, run_id="sarue", validation_set=(x_weeb, y_weeb), n_epoch=200)
    
    score = model.evaluate(x_weeb, y_weeb)
    print("Result: {}".format(score[0]))
    
    results = model.predict(x_weeb)
    score = mean_absolute_error(y_weeb, results)
    print("Kaggle score: {}".format(score))
    
    # Agora gerar output
    results = model.predict(x_data_te.values)
    with open("output.csv", "w") as weeb:
        weeb.write("Id,production\n")
        for id, result in zip(id_data_te, results):
            weeb.write(str(id) + "," + str(result[0]) + "\n")

if __name__ == "__main__":
    main()

# ==============================================================================
