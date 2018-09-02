#======================================================================================================================#
#   class Field
#----------------------------------------------------------------------------------------------------------------------#
#   :author: Rafael Toyomoto
#======================================================================================================================#

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

import utils.parser as dataset

#----------------------------------------------------------------------------------------------------------------------#


DATASET_DIR = "dataset/"


#----------------------------------------------------------------------------------------------------------------------#

df_fields = dataset.get_soil_data(full=True)
df_train = dataset.get_production()

df_full = df_train.merge(df_fields, how="inner", left_on=["field", "date"], right_on=["field", "date"])


correlation = df_fields[["BLDFIE_sl1", "BLDFIE_sl2", "BLDFIE_sl3", "BLDFIE_sl4", "BLDFIE_sl5", "BLDFIE_sl6", "BLDFIE_sl7"]].corr()
sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
plt.show()


#sns.lineplot(x="date", y="production", data=df_train)
#plt.show()

#train_data = pd.read_csv(DATASET_DIR + TRAIN_FILENAME)

#sns.lineplot(x='date', y="production", data=train_data)
#plt.show()

#def generate_dates(dataframe, year_str, month_str):
#    dataframe["date"] = pd.to_datetime((dataframe[year_str] * 10000 + dataframe[month_str] * 100 + 1).apply(str), format="%Y%m%d")

#sns.barplot(x='type', y='production', data=train_data)
#plt.show()

#join_field = train_data.merge(full_field_data, how="inner", left_on=["field", "date"], right_on=["field", "date"])


#======================================================================================================================#
