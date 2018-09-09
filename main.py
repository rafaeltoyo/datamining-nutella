# ======================================================================================================================
#   class Field
# ----------------------------------------------------------------------------------------------------------------------
#   :author: Rafael Toyomoto
# ======================================================================================================================

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

import utils.parser as dataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------------------------------------------------------

df_fields = dataset.get_soil_data(full=True)
df_train = dataset.get_production()

df_full = df_train.merge(df_fields, how="inner", left_on=["field", "date"], right_on=["field", "date"])


def tatica_1():
    for label in ["BLDFIE", "CECSOL", "CLYPPT", "CRFVOL", "ORCDRC", "PHIHOX", "PHIKCL", "SLTPPT", "SNDPPT"]:
        correlation = df_fields[
            [label + "_sl1", label + "_sl2", label + "_sl3", label + "_sl4", label + "_sl5", label + "_sl6",
             label + "_sl7"]].corr()
        sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Reds", vmax=1.0, vmin=-1.0)
        plt.show()

    correlation = df_fields[["OCSTHA_sd1", "OCSTHA_sd2", "OCSTHA_sd3", "OCSTHA_sd4", "OCSTHA_sd5", "OCSTHA_sd6"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Reds", vmax=1.0, vmin=-1.0)
    plt.show()


def tatica_pca():
    df = pd.DataFrame(data=df_full,
                      columns=["CLYPPT_sl1", "CLYPPT_sl2", "CLYPPT_sl3", "CLYPPT_sl4", "CLYPPT_sl5", "CLYPPT_sl6",
                               "CLYPPT_sl7"])

    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)

    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)

    plt.scatter(x_pca[:, 0], x_pca[:, 1], cmap='plasma')
    plt.show()


def tatica_1_fim():
    correlation = df_fields[
        ["BLDFIE_sl1", "BLDFIE_sl3", "BLDFIE_sl6", "CECSOL_sl3", "CLYPPT_sl2", "CLYPPT_sl6", "CRFVOL_sl4", "OCSTHA_sd1",
         "OCSTHA_sd2", "OCSTHA_sd6", "ORCDRC_sl1", "ORCDRC_sl2", "ORCDRC_sl7", "PHIHOX_sl3", "PHIKCL_sl4", "SLTPPT_sl6",
         "SNDPPT_sl4", "production"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Reds", vmax=1.0, vmin=-1.0)
    plt.show()


def tatica_2():
    sns.lineplot(x='date', y="production", data=df_full, palette="Blues", linewidth=2.5)
    plt.show()
    sns.lineplot(x='month', y="production", data=df_full, palette="Blues", linewidth=2.5)
    plt.show()


sns.barplot(x='field', y="production", data=df_full)
plt.show()
sns.barplot(x='type', y="production", data=df_full)
plt.show()
sns.barplot(x='age', y="production", data=df_full)
plt.show()

# ======================================================================================================================
