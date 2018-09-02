#======================================================================================================================#
#   class Field
#----------------------------------------------------------------------------------------------------------------------#
#   :author: Rafael Toyomoto
#======================================================================================================================#

import pandas as pd

#----------------------------------------------------------------------------------------------------------------------#

def get_production():
    out = pd.read_csv("dataset/train.csv")
    out["date"] = pd.to_datetime((out["harvest_year"] * 10000 + out["harvest_month"] * 100 + 1).apply(str),format="%Y%m%d")
    return out

def get_soil_data(full=False):
    out = pd.read_csv("dataset/soil_data.csv")
    # Add History
    if full:
        data = []
        for iter in range(0,27):
            data.insert(iter, get_field_info(iter))
            data[iter]["field"] = iter
        join = pd.concat(data)
        out = join.merge(out, how="inner", left_on=["field"], right_on=["field"])
    return out

def get_field_info(number):
    out = pd.read_csv("dataset/field-" + str(number) + ".csv")
    out["date"] = pd.to_datetime((out["year"] * 10000 + out["month"] * 100 + 1).apply(str),format="%Y%m%d")
    return out

#======================================================================================================================#