# -*- coding: utf-8 -*-

# ==============================================================================
# A melhor loli está aqui!
# ==============================================================================

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import linregress

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

import sklearn
import sklearn.cluster
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN

# ==============================================================================

DATASET_DIR = "../dataset/"

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
SOIL_FILENAME = "soil_data.csv"

FIELD_FILENAMES = [("field-%d.csv" % i) for i in range(28)]

WEATHER_ATTRIBUTES = [
    'acc_precipitation',
    'month',
    'temperature',
    #'windspeed',
    #'type',
    #'age',
    #'timestamp',
    # 'prod_error',
    # 'prod_stddev',
    #'beh_type_field'
    'production'
]
SOIL_ATTRIBUTES = [
    "BDRLOG_BDRLOG_M",
    # "BDTICM_BDTICM_M",
    # "BLDFIE_sl1",
    # "BLDFIE_sl3",
    # "BLDFIE_sl7",
    # "CECSOL_sl4",
    # "CLYPPT_sl2",
    # "CLYPPT_sl6",
    # "CRFVOL_sl4",
    # "OCSTHA_sd4",
    # "ORCDRC_sl1",
    # "ORCDRC_sl3",
    # "ORCDRC_sl6",
    # "PHIHOX_sl4",
    # "PHIKCL_sl4",
    # "SLTPPT_sl4",
    # "SNDPPT_sl4",
    'prod_mean'
]

PCA_ATTRIBUTES = 3

DATASET = 2
ALGO = 1

# AggCl
N_CLUSTER = 3
LINKAGE = 'average'  # ward", "complete", "average"

# DBSCAN
NS_DBSCAN = 30
EP_DBSCAN = 0.175

NORMALIZE = True


# ==============================================================================


def main():
    train_data = pd.read_csv(DATASET_DIR + TRAIN_FILENAME)
    test_data = pd.read_csv(DATASET_DIR + TEST_FILENAME)
    soil_data = pd.read_csv(DATASET_DIR + SOIL_FILENAME)
    full_dataset = pd.read_csv("monolith.csv")
    test_dataset = pd.read_csv("monolith_test.csv")

    # Carrega os dados
    if DATASET == 1:

        for field in range(28):
            soil_data.loc[soil_data["field"] == field, "prod_mean"] = train_data.loc[
                train_data["field"] == field, "production"].mean()
            soil_data.loc[soil_data["field"] == field, "prod_stddev"] = train_data.loc[
                train_data["field"] == field, "production"].std()

        data = soil_data[["field"] + SOIL_ATTRIBUTES].copy()

        FULL_ATTRIBUTES = SOIL_ATTRIBUTES

    else:
        grouped = full_dataset.groupby(['type', 'field'])
        df = pd.DataFrame(grouped.apply(lambda x: linregress(x['month'], x['production'])))
        df[0].apply(pd.Series)
        #df[['slope', 'intercept', 'r_value', 'p_value', 'std_err']] = df[0].apply(pd.Series)
        #del df[0]
        iter = 0
        for (type, field) in zip(*df[0].keys().labels):
            valid_values = (full_dataset['type'] == type) & (full_dataset['field'] == field)
            full_dataset.loc[valid_values, 'beh_type_field'] = float(df[0].get([iter]).apply(pd.Series)[0][0])
            iter += 1

        #full_dataset['beh_type_field'] -= full_dataset['beh_type_field'].min()

        for type in range(0, 7):
            for month in range(1, 13):
                valid_values = (full_dataset['type'] == type) & (full_dataset['month'] == month)
                full_dataset.loc[valid_values, 'type_prod_mean'] = full_dataset.loc[valid_values, 'production'].mean()

        for field in range(28):
            full_dataset.loc[full_dataset["field"] == field, "prod_mean"] = train_data.loc[
                train_data["field"] == field, "production"].mean()
            full_dataset.loc[full_dataset["field"] == field, "prod_stddev"] = train_data.loc[
                train_data["field"] == field, "production"].std()

        full_dataset['timestamp'] = full_dataset['year'] * 12 + full_dataset['month']


        # full_dataset['prod_error'] = (full_dataset['production'] - full_dataset['production'].mean()) ** 2
        # full_dataset.loc[full_dataset['prod_error'] > 0.2, 'prod_error'] = 0.2

        data = full_dataset[["field"] + WEATHER_ATTRIBUTES].copy()

        FULL_ATTRIBUTES = WEATHER_ATTRIBUTES

    if NORMALIZE:
        x = data.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        data = pd.DataFrame(x_scaled, columns=data.columns, index=data.index)

    # ==============================================================================

    if ALGO == 1:
        # Agglomerative Clustering
        sarue = sklearn.cluster.AgglomerativeClustering(n_clusters=N_CLUSTER, linkage=LINKAGE)
        y_clusters = sarue.fit_predict(data)

        n_clusters = N_CLUSTER

    else:
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=NS_DBSCAN).fit(data)
        distances, indices = nbrs.kneighbors(data)
        distanceDec = sorted(distances[:, NS_DBSCAN - 1], reverse=True)
        plt.plot(indices[:, 0], distanceDec)
        plt.show()

        db = DBSCAN(eps=EP_DBSCAN, min_samples=NS_DBSCAN)
        y_clusters = db.fit_predict(data)

        data['labels'] = db.labels_
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        cols = sns.color_palette("Set2", n_colors=n_clusters, desat=.5)
        #y_clusters = [cols[i] for i in db.labels_]

    # ==============================================================================

    if len(FULL_ATTRIBUTES) == 2:
        plt.scatter(data[FULL_ATTRIBUTES[0]], data[FULL_ATTRIBUTES[1]], c=y_clusters, cmap="viridis")
        plt.title('Number of components: ' + str(n_clusters))
        plt.xlabel(FULL_ATTRIBUTES[0])
        plt.ylabel(FULL_ATTRIBUTES[1])
        plt.show()

    elif PCA_ATTRIBUTES == 2:
        pca = sklearn.decomposition.PCA(n_components=2)
        principal_components = pca.fit_transform(data[FULL_ATTRIBUTES])
        principal_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

        plt.scatter(principal_df["PC1"], principal_df["PC2"], c=y_clusters, cmap="viridis")
        plt.title('Number of components: ' + str(n_clusters))
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

    elif len(FULL_ATTRIBUTES) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[FULL_ATTRIBUTES[0]], data[FULL_ATTRIBUTES[1]], data[FULL_ATTRIBUTES[2]], c=y_clusters,
                   cmap="viridis")
        ax.set_title('Number of components: ' + str(n_clusters))
        ax.set_xlabel(FULL_ATTRIBUTES[0])
        ax.set_ylabel(FULL_ATTRIBUTES[1])
        ax.set_zlabel(FULL_ATTRIBUTES[2])
        plt.show()

    elif PCA_ATTRIBUTES == 3:
        pca = sklearn.decomposition.PCA(n_components=3)
        principal_components = pca.fit_transform(data[FULL_ATTRIBUTES])
        principal_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2", "PC3"])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(principal_df["PC1"], principal_df["PC2"], principal_df["PC3"], c=y_clusters, cmap="viridis")
        ax.set_title('Number of components: ' + str(n_clusters))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.show()


if __name__ == "__main__":
    main()

# ==============================================================================
