# -*- coding: utf-8 -*-

# ==============================================================================
# Comentários, etc, etc...
# ==============================================================================

import pandas as pd
import numpy as np
import sklearn.svm

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ==============================================================================

DATASET_FILENAME = "monolith.csv"
TEST_FILENAME = "monolith_test.csv"

ATTRIBUTES = [
    "acc_precipitation",
    "age", # Se tirar isso aqui fica uma porcaria
    "temperature",
    "windspeed",
    #"dewpoint",
    "Soilwater_L1",
    #"Soilwater_L2",
    #"Soilwater_L3",
    #"Soilwater_L4",
    "month",
    #"BLDFIE_sl7", # Se colocar esse aqui fica uma porcaria
    #"CRFVOL_sl4",
    #"ORCDRC_sl6", # Erhm
    #"BDRLOG_BDRLOG_M", # Erhm
    #"CECSOL_sl4",
    #"CLYPPT_sl6",
]
    
# ==============================================================================

def main():
    train_dataset = pd.read_csv(DATASET_FILENAME)
    test_dataset = pd.read_csv(TEST_FILENAME)
    
    full_dataset = pd.concat([train_dataset, test_dataset])
    for attribute in ["age"]:
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
    
    best_score = -1
    best_svr = None
    best_x_test = None
    best_y_test = None
    best_results = None
    
    for _ in range(50):
        x_train, x_test, y_train, y_test = train_test_split(x_data_tr, y_data_tr,
                test_size=0.2)
        svr = sklearn.svm.SVR(kernel="rbf")
        svr.fit(x_train, y_train)
        results = svr.predict(x_test)
        score = svr.score(x_test, y_test)
        print(score)
        if score > best_score:
            best_score = score
            best_svr = svr
            best_y_test = y_test
            best_results = results
    
    #print(best_score)
    mae = 0
    for item_y, item_r in zip(best_y_test, best_results):
        #print(item_y, item_r)
        mae += abs(item_y - item_r)
    mae /= len(best_y_test)
    print("Kaggle score: " + str(mae))
    
    # Agora gerar output
    
    #min_max_scaler = preprocessing.MinMaxScaler()
    #np_scaled = min_max_scaler.fit_transform(x_data_te)
    #x_data_te = pd.DataFrame(np_scaled)
    
    results = best_svr.predict(x_data_te)
    
    with open("output.csv", "w") as weeb:
        for id, result in zip(id_data_te, results):
            weeb.write(str(id) + "," + str(result) + "\n")

if __name__ == "__main__":
    main()

# ==============================================================================
