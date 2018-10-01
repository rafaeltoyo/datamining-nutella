# -*- coding: utf-8 -*-

# ==============================================================================
# Comentários, etc, etc...
# ==============================================================================

import pandas as pd
import numpy as np
import tflearn

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression

from xgboost import XGBRegressor

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
    "Soilwater_L4",
    #"month",
    #"year",
    #"Precipitation",
    #"field",
    "mean_prod_mon",
    "mean_prod_field",
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

    for month in range(1, 13):
        avg_prod = train_dataset.loc[train_dataset["month"] == month, "production"].mean()
        train_dataset.loc[train_dataset["month"] == month, "mean_prod_mon"] = avg_prod
        test_dataset.loc[test_dataset["month"] == month, "mean_prod_mon"] = avg_prod
    
    for field in range(28):
        avg_prod = train_dataset.loc[train_dataset["field"] == field, "production"].mean()
        train_dataset.loc[train_dataset["field"] == field, "mean_prod_field"] = avg_prod
        test_dataset.loc[test_dataset["field"] == field, "mean_prod_field"] = avg_prod
    
    
    full_dataset = pd.concat([train_dataset, test_dataset])
    for attribute in ATTRIBUTES:
        max_value = full_dataset[attribute].max()
        min_value = full_dataset[attribute].min()
        train_dataset[attribute] = (train_dataset[attribute] - min_value) / (max_value - min_value)
        test_dataset[attribute] = (test_dataset[attribute] - min_value) / (max_value - min_value)
    
        
    # Modelo 1: tipos 0, 5 e 6
    
    """
    train_model_1 = train_dataset[train_dataset["type"].isin([0, 5, 6])]
    test_model_1 = test_dataset[test_dataset["type"].isin([0, 5, 6])]
    urf = EllipticEnvelope(contamination=0.07)
    urf.fit(train_model_1[ATTRIBUTES].values.reshape(-1, len(ATTRIBUTES)))
    train_model_1["outlier"] = urf.predict(train_model_1[ATTRIBUTES].values.reshape(-1, len(ATTRIBUTES)))
    train_model_1 = train_model_1[train_model_1["outlier"] == 1]
    train_model_1.drop("outlier", axis=1)
    
    x_train_1 = train_model_1[ATTRIBUTES]
    y_train_1 = train_model_1["production"]
    
    x_test_1 = test_model_1[ATTRIBUTES]
    id_test_1 = test_model_1["Id"]
    
    model_1 = AdaBoostRegressor(base_estimator=XGBRegressor(max_depth=7, learning_rate=0.05, n_estimators=100, n_jobs=8, base_score=0.05), n_estimators=75, learning_rate=1, loss="exponential")
    #model_1 = RandomForestRegressor()
    scores = cross_val_score(model_1, x_train_1, y_train_1, cv=10, scoring="neg_mean_absolute_error")
    print("Kaggle score (modelo 1):")
    print(scores)
    print()
    
    model_1.fit(x_train_1, y_train_1)
    results_1 = model_1.predict(x_test_1)
    
    
    # Modelo 2: tipos 1, 2 e 4
    
    train_model_2 = train_dataset[train_dataset["type"].isin([1, 2, 4])]
    test_model_2 = test_dataset[test_dataset["type"].isin([1, 2, 4])]
    urf = EllipticEnvelope(contamination=0.16)
    urf.fit(train_model_2[ATTRIBUTES].values.reshape(-1, len(ATTRIBUTES)))
    train_model_2["outlier"] = urf.predict(train_model_2[ATTRIBUTES].values.reshape(-1, len(ATTRIBUTES)))
    train_model_2 = train_model_2[train_model_2["outlier"] == 1]
    train_model_2.drop("outlier", axis=1)
    
    x_train_2 = train_model_2[ATTRIBUTES]
    y_train_2 = train_model_2["production"]
    
    x_test_2 = test_model_2[ATTRIBUTES]
    id_test_2 = test_model_2["Id"]
    
    model_2 = AdaBoostRegressor(base_estimator=XGBRegressor(max_depth=7, learning_rate=0.05, n_estimators=100, n_jobs=8, base_score=0.05), n_estimators=75, learning_rate=1, loss="exponential")
    #model_2 = RandomForestRegressor()
    scores = cross_val_score(model_2, x_train_2, y_train_2, cv=10, scoring="neg_mean_absolute_error")
    print("Kaggle score (modelo 2):")
    print(scores)
    print()
    
    model_2.fit(x_train_2, y_train_2)
    results_2 = model_2.predict(x_test_2)
    
    
    # Modelo 3: tudo
    
    train_model_3 = train_dataset
    test_model_3 = test_dataset[test_dataset["type"].isin([-1, 3, 7])]
    urf = EllipticEnvelope(contamination=0.06)
    urf.fit(train_model_3[ATTRIBUTES].values.reshape(-1, len(ATTRIBUTES)))
    train_model_3["outlier"] = urf.predict(train_model_3[ATTRIBUTES].values.reshape(-1, len(ATTRIBUTES)))
    train_model_3 = train_model_3[train_model_3["outlier"] == 1]
    train_model_3.drop("outlier", axis=1)
    
    x_train_3 = train_model_3[ATTRIBUTES]
    y_train_3 = train_model_3["production"]
    
    x_test_3 = test_model_3[ATTRIBUTES]
    id_test_3 = test_model_3["Id"]
    
    model_3 = AdaBoostRegressor(base_estimator=XGBRegressor(max_depth=7, learning_rate=0.05, n_estimators=100, n_jobs=8, base_score=0.05), n_estimators=75, learning_rate=1, loss="exponential")
    #model_3 = RandomForestRegressor()
    scores = cross_val_score(model_3, x_train_3, y_train_3, cv=10, scoring="neg_mean_absolute_error")
    print("Kaggle score (modelo 3):")
    print(scores)
    print()
    
    model_3.fit(x_train_3, y_train_3)
    results_3 = model_3.predict(x_test_3)
    
    # Agora gerar output
    with open("output.csv", "w") as weeb:
        weeb.write("Id,production\n")
        for id, result in zip(id_test_1, results_1):
            weeb.write(str(id) + "," + str(result) + "\n")
        for id, result in zip(id_test_2, results_2):
            weeb.write(str(id) + "," + str(result) + "\n")
        for id, result in zip(id_test_3, results_3):
            weeb.write(str(id) + "," + str(result) + "\n")
    """
    
    """
    train_filtered = []
    for field in range(28):
        noob = train_dataset[train_dataset["field"] == field]
        #train_filtered.append(noob[noob["production"] < noob["production"].quantile(0.90)])
        urf = EllipticEnvelope(contamination=0.06)
        urf.fit(noob["production"].values.reshape(-1, 1))
        noob["outlier"] = urf.predict(noob["production"].values.reshape(-1, 1))
        print(noob["outlier"].value_counts())
        noob = noob[noob["outlier"] == 1] # Pega só os inliers
        noob.drop("outlier", axis=1)
        train_filtered.append(noob)
    train_dataset = pd.concat(train_filtered)
    """
    
    urf = EllipticEnvelope(contamination=0.08)
    urf.fit(train_dataset["production"].values.reshape(-1, 1))
    train_dataset["outlier"] = urf.predict(train_dataset["production"].values.reshape(-1, 1))
    #print(train_dataset["outlier"].value_counts())
    train_dataset = train_dataset[train_dataset["outlier"] == 1]
    train_dataset.drop("outlier", axis=1)
    
    
    x_data_tr = train_dataset[ATTRIBUTES]
    y_data_tr = train_dataset["production"]
    
    x_data_te = test_dataset[ATTRIBUTES]
    id_data_te = test_dataset["Id"]

    # SPLIT DATASET
    #x_train, x_test, y_train, y_test = train_test_split(
    #        x_data_tr, y_data_tr, test_size=0.2)
    
    # FULL DATASET
    x_train = x_data_tr
    x_test = x_data_tr
    y_train = y_data_tr
    y_test = y_data_tr
    
    
    
    network = input_data(shape=[None, len(ATTRIBUTES)], name="Input_layer")
    #network = fully_connected(network, 24, activation="relu", name="Hidden_layer_1")
    network = fully_connected(network, 20, activation="relu", name="Hidden_layer_2")
    network = fully_connected(network, 16, activation="relu", name="Hidden_layer_3")
    network = fully_connected(network, 12, activation="relu", name="Hidden_layer_4")
    network = fully_connected(network, 1, activation="linear", name="Output_layer")
    network = regression(network, batch_size=64, optimizer='adam', learning_rate=0.001, loss="mean_square", metric="R2")
    
    model = tflearn.DNN(network)
    
    
    
    x_sarue = x_train.values
    y_sarue = y_train.values.reshape(-1, 1)
    x_weeb = x_test.values
    y_weeb = y_test.values.reshape(-1, 1)
    
    
    
    #model.fit(x_sarue, y_sarue, show_metric=True, run_id="sarue", validation_set=(x_weeb, y_weeb), n_epoch=200)
    model.fit(x_sarue, y_sarue, show_metric=True, run_id="weeb", validation_set=0.2, n_epoch=500)
    
    score = model.evaluate(x_weeb, y_weeb)
    print("Result: {}".format(score[0]))
    
    
    """
    #model = AdaBoostRegressor(n_estimators=75, learning_rate=1.0, loss="square")
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    
    #results = model.predict(x_weeb)
    #score = mean_absolute_error(y_weeb, results)
    results = model.predict(x_test)
    score = mean_absolute_error(y_test, results)
    print("Kaggle score: {}".format(score))
    """
    
    
    # Agora gerar output
    results = model.predict(x_data_te.values)
    with open("output.csv", "w") as weeb:
        weeb.write("Id,production\n")
        for id, result in zip(id_data_te, results):
            weeb.write(str(id) + "," + str(result[0]) + "\n")
            #weeb.write(str(id) + "," + str(result) + "\n")
    

if __name__ == "__main__":
    main()

# ==============================================================================
