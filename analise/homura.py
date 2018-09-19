# -*- coding: utf-8 -*-

# ==============================================================================
# A melhor loli está aqui!
# ==============================================================================

import numpy as np
import pandas as pd
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

import sklearn
import sklearn.cluster
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN

# ==============================================================================

DATASET_DIR = "../dataset/"

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
SOIL_FILENAME = "soil_data.csv"

FIELD_FILENAMES = [("field-%d.csv" % i) for i in range(28)]


# ==============================================================================


def main():
    """
    Função Main
    :return:
    """

    PCA_ATTRIBUTES = 3
    """
    FULL_ATTRIBUTES = ["BDRLOG_BDRLOG_M", "BDTICM_BDTICM_M", "BLDFIE_sl1", "BLDFIE_sl3", "BLDFIE_sl7", "CECSOL_sl4",
                       "CLYPPT_sl2", "CLYPPT_sl6", "CRFVOL_sl4", "OCSTHA_sd4", "ORCDRC_sl1", "ORCDRC_sl3", "ORCDRC_sl6",
                       "PHIHOX_sl4", "PHIKCL_sl4", "SLTPPT_sl4", "SNDPPT_sl4", "prod_mean"]
    """
    FULL_ATTRIBUTES = ["prod_mean", "temperature", "windspeed", "dewpoint", "acc_precipitation", "month"]

    # Carrega os dados
    full_dataset = pd.read_csv("monolith.csv")
    test_dataset = pd.read_csv("monolith_test.csv")
    train_data = pd.read_csv(DATASET_DIR + TRAIN_FILENAME)
    test_data = pd.read_csv(DATASET_DIR + TEST_FILENAME)
    soil_data = pd.read_csv(DATASET_DIR + SOIL_FILENAME)

    for field in range(28):
        soil_data.loc[soil_data["field"] == field, "prod_mean"] = train_data.loc[
            train_data["field"] == field, "production"].mean()
        soil_data.loc[soil_data["field"] == field, "prod_stddev"] = train_data.loc[
            train_data["field"] == field, "production"].std()
        full_dataset.loc[full_dataset["field"] == field, "prod_mean"] = full_dataset.loc[
            full_dataset["field"] == field, "production"].mean()
        full_dataset.loc[full_dataset["field"] == field, "prod_stddev"] = full_dataset.loc[
            full_dataset["field"] == field, "production"].std()

    data = full_dataset[["field"] + FULL_ATTRIBUTES].copy()

    # K-means
    kmeans = KMeans(n_clusters=3, init="random")
    kmeans.fit(data)
    y_kmeans = kmeans.predict(data)

    # Agglomerative Clustering
    sarue = sklearn.cluster.AgglomerativeClustering(n_clusters=3, linkage="ward")
    y_clusters = sarue.fit_predict(data)

    if PCA_ATTRIBUTES == 2:
        pca = sklearn.decomposition.PCA(n_components=2)
        principal_components = pca.fit_transform(data[FULL_ATTRIBUTES])
        principal_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

        plt.scatter(principal_df["PC1"], principal_df["PC2"], c=y_clusters, cmap="viridis")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

    elif PCA_ATTRIBUTES == 3:
        pca = sklearn.decomposition.PCA(n_components=3)
        principal_components = pca.fit_transform(data[FULL_ATTRIBUTES])
        principal_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2", "PC3"])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(principal_df["PC1"], principal_df["PC2"], principal_df["PC3"], c=y_clusters, cmap="viridis")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.show()

    # features_to_compute = ["temperature", "windspeed", "Precipitation", "acc_precipitation", "production"]
    # quero = full_dataset[full_dataset['type'].isin(['4'])]

    cluster_1 = False
    cluster_2 = False

    # ================================================================================================================ #
    if cluster_1:
        """
        Condições climáticas:
            Dados agregados para tentar gerar clusters
            - temperature
            - acc_precipitation
            - windspeed
            Classificação objetivada: 
            - month
        """

        # Gerar Figura para plot 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plotar pontos
        ax.scatter(full_dataset['temperature'], full_dataset['acc_precipitation'], full_dataset['windspeed'],
                   c=full_dataset['cat_month'].cat.codes, cmap="viridis")
        ax.view_init(30, 185)
        plt.show()

        """
        Realizar o agrupamento dos dados de condições climáticas.
        Como observado no gráfico anterior, clusters por semestre ou trimestre pode ser uma alternativa (2-3 clusters)
        Táticas:
            K-means
            Agglomerative Clustering
        """

        x1 = np.array(full_dataset['temperature'])
        x2 = np.array(full_dataset['acc_precipitation'], )
        x3 = np.array(full_dataset['windspeed'])

        # create new plot and data
        plt.plot()
        X = np.array(list(zip(x1, x2, x3))).reshape(len(x1), 3)

        # k means determine k
        distortions = []
        K = range(1, 10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k).fit(X)
            kmeanModel.fit(X)
            distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

        # Plot the elbow
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

        weather_data = full_dataset[["temperature", "acc_precipitation", "windspeed"]]
        # K-means
        kmeans = KMeans(n_clusters=3, init="random")
        kmeans.fit(weather_data)
        y_kmeans = kmeans.predict(weather_data)

        # Agglomerative Clustering
        sarue = sklearn.cluster.AgglomerativeClustering(n_clusters=3, linkage="ward")
        y_clusters = sarue.fit_predict(weather_data)

        # Gerar Figura para plot 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plotar pontos
        ax.scatter(full_dataset["temperature"], full_dataset["acc_precipitation"], full_dataset["windspeed"],
                   c=y_kmeans, cmap="viridis")
        ax.view_init(30, 185)
        plt.show()

        # Gerar Figura para plot 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plotar pontos
        ax.scatter(full_dataset["temperature"], full_dataset["acc_precipitation"], full_dataset["windspeed"],
                   c=y_clusters, cmap="viridis")
        ax.view_init(30, 185)
        plt.show()

    # ================================================================================================================ #
    if cluster_2:
        """
        Produções por idade e tipo ao longo dos anos:
            Dados para tentar gerar clusters
            - age
            - timestamp (ano + mes)
            - field
            Classificação objetivada: 
            - type
        """

        # Tentar algo com producao
        quero = full_dataset[full_dataset['type'].isin(['0', '1', '2', '3', '4', '5', '6'])]
        quero['type'] = pd.Categorical(quero['type'])
        quero['timestamp'] = quero['year'] * 12 + quero['month']
        quero = quero[['age', 'timestamp', 'field', 'type']]

        # Gerar Figura para plot 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plotar pontos
        ax.scatter(quero['age'], quero['timestamp'], quero['type'], c=quero['type'].cat.codes, cmap="viridis")
        ax.view_init(30, 185)
        plt.show()

        """
        O gráfico apresenta 'linhas' de possíveis levas de árvores plantadas juntas e mantidas ao longo dos anos.
        """

        data = quero[['age', 'timestamp', 'type']]

        ns = 10
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=ns).fit(data)
        distances, indices = nbrs.kneighbors(data)
        distanceDec = sorted(distances[:, ns - 1], reverse=True)
        plt.plot(indices[:, 0], distanceDec)
        plt.show()
        # ns = 5 -> epsilon = 1.2 - 1.5
        epsilon = 2

        """
        DB scan
            n = 10
            ep = 1.5
        """

        db = DBSCAN(eps=epsilon, min_samples=ns)

        # Fit and get labels
        db.fit_predict(data)
        data['labels'] = db.labels_
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

        # Plot results
        cols = sns.color_palette("Set2", n_colors=n_clusters, desat=.5)
        cl = [cols[i] for i in db.labels_]

        # Gerar Figura para plot 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data['age'], data['timestamp'], data['type'], c=cl, alpha=0.5)
        ax.set_title('Number of components: ' + str(n_clusters))
        ax.set_xlabel('Age')
        ax.set_ylabel('Timestamp (year * 12 + month)')
        ax.set_zlabel('Field')

        # Show aggregated volume and interest at each neighborhood
        """
        x = data.groupby('labels')[['Age', 'Timestamp', 'Field']].mean().sort_values(['response'])
        x = pd.concat([x, data['labels'].value_counts()], axis=1).sort_values(['response'])
        cols = sns.color_palette("RdBu_r", n_clusters)[::-1]
        for i in range(n_clusters):
            props = dict(boxstyle='round', facecolor=cols[i], alpha=0.8)
            ax.text(x.longitude.values[i], x.latitude.values[i],
                        str(np.array(np.round(x.response.values, 2), '|S8')[i]) + '\n' + str(
                            np.array(x['labels'].values, '|S8')[i]),
                        fontsize=9, verticalalignment='center', horizontalalignment='center', bbox=props)
        """
        plt.show()


if __name__ == "__main__":
    main()

# ==============================================================================
