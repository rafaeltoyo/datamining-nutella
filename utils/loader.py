#======================================================================================================================#
#   Open and parse dataset .csv files
#----------------------------------------------------------------------------------------------------------------------#
#   :author: Rafael Toyomoto
#======================================================================================================================#

from os import walk
import csv
import re

from entity.field import Field
from entity.history import History


def open_csv(csvfile, delimiter=','):

    # Decode the csv file
    reader = csv.reader(csvfile, delimiter=delimiter)

    output = []

    header = None
    for row in reader:

        if header is None:

            # get the csv header
            header = []
            iter = 0
            for label in row:
                header.insert(iter, label)
                iter += 1

        else:

            # save a data row
            data = {}
            iter = 0
            for elem in row:
                data[header[iter]] = elem
                iter += 1
            output.append(data)

    return output


def load_csv(filename, path="dataset/"):

    # Get file content and decode that as a csv file
    with open(path + filename) as file:

        return open_csv(file)


def load_fields():

    fields = load_csv("soil_data.csv")
    for row in fields:
        field = Field.export(row)


def loader(path="dataset/"):

    # Open all files in path
    for (dirpath, dirnames, filenames) in walk(path):
        for filename in filenames:

            # Get all csv files
            if re.search('[a-zA-Z0-9\-_]*\.csv$', filename) is not None:

                # Open the .csv file
                yield load_csv(filename, path=path)
