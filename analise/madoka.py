# -*- coding: utf-8 -*-

# ==================================================================================================================== #
# Author: Rafael Hideo Toyomoto
# ==================================================================================================================== #

import math
import time

import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


# ==================================================================================================================== #


def main():
    # Train dataset
    full_dataset = pd.read_csv("monolith.csv")
    #['date'] = time.mktime(time.strptime(full_dataset['date'], '%Y-%m-%d'))
    full_dataset['date'] = full_dataset['year'] * 12 + full_dataset['month']
    # Test dataset
    test_dataset = pd.read_csv("monolith_test.csv")
    test_dataset['date'] = test_dataset['year'] * 12 + test_dataset['month']
    #test_dataset['date'] = time.mktime(time.strptime(test_dataset['date'], '%Y-%m-%d'))

    """
    Prepare Model
    """

    # This line instantiates the model.
    rf = RandomForestRegressor()
    df = full_dataset[~full_dataset.production.isnull()]
    X_train = full_dataset.drop(['production', 'Id'], axis=1)
    y_train = full_dataset.production.values

    # Fit the model on your training data.
    rf.fit(X_train, y_train)

    # Feature Importances
    feature_importances = pd.DataFrame(rf.feature_importances_, index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False).reset_index()

    # Prepare
    df_train = full_dataset[~full_dataset.production.isnull()]
    X = df_train.drop(['production', 'Id'], axis=1)

    # Filter importance
    features = list(feature_importances['index'].values)[:50]
    X = X[features]
    y = df.production.values

    # normalize
    scaler = StandardScaler()
    norm_X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(norm_X, y, test_size=0.2,
                                                                                random_state=1)
    (X_train.shape, X_test.shape)

    base_model = RandomForestRegressor()
    base_model.fit(X_train, y_train)

    y_hat = base_model.predict(X_test)

    score_mae = sklearn.metrics.mean_absolute_error(y_test, y_hat)
    r2 = sklearn.metrics.r2_score(y_test, y_hat)

    print(score_mae)
    print(r2)

    """
    Submission
    """

    ## Filter importance
    X = test_dataset[features]
    scaler = StandardScaler()
    norm_X = scaler.fit_transform(X)
    X = scaler.transform(X)  # normalize

    # y
    y = df.production.values

    prod = base_model.predict(X)
    # print(prod[:10])

    f = open('submission.csv', 'w')
    f.write("Id,production\n")
    for i in range(len(test_dataset.Id.values)):
        _id = test_dataset.Id.values[i]
        p = math.fabs(prod[i])
        f.write("{},{}\n".format(_id, p))
    f.close()


# ==================================================================================================================== #


if __name__ == "__main__":
    main()

# ==================================================================================================================== #
