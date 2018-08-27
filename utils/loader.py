#==============================================================================#
#   Parser do Dataset
#==============================================================================#

from os import walk
import csv
import re

class Loader(object):
    def __init__(self, path="../dataset/"):
        self.path = path
    
    def __explore_dataset_dir(self):
        f = []
        for (dirpath, dirnames, filenames) in walk(self.path):
            f.extend(filenames)
            return f

    def __read_csv(self, filename):
        firstLine = True
        with open(self.path + filename, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                if firstLine:
                    firstLine = False
                else:
                    yield row

    def load_fields(self):
        f = self.__explore_dataset_dir()
        if f is None:
            return
        for filename in f:
            if re.search("^field-[0-9]{1,2}\.csv$", filename) is not None:
                for row in self.__read_csv(filename):
                    yield row

