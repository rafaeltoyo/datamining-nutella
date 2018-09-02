#======================================================================================================================#
#   class History
#----------------------------------------------------------------------------------------------------------------------#
#   :author: Rafael Toyomoto
#======================================================================================================================#

class History(object):

    def __init__(self, row=None):

        self.timespam = 0
        self.month = 0
        self.year = 0
        self.temperature = 0
        self.dewpoint = 0
        self.windspeed = 0
        self.soilwater_l1 = 0
        self.soilwater_l2 = 0
        self.soilwater_l3 = 0
        self.soilwater_l4 = 0
        self.precipitation = 0

    @staticmethod
    def export(row):

        o = History()

        o.timespam = int(row['month']) + (int(row['year']) - 2000) * 12
        o.month = int(row['month'])
        o.year = int(row['year'])
        o.temperature = float(row['temperature'])
        o.dewpoint = float(row['dewpoint'])
        o.windspeed = float(row['windspeed'])
        o.soilwater_l1 = float(row['Soilwater_L1'])
        o.soilwater_l2 = float(row['Soilwater_L2'])
        o.soilwater_l3 = float(row['Soilwater_L3'])
        o.soilwater_l4 = float(row['Soilwater_L4'])
        o.precipitation = float(row['Precipitation'])
