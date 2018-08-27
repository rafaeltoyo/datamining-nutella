#==============================================================================#
#==============================================================================#

class Field(object):
    def __init__(self, row=None):
        if row is not None:
            self.month = int(row[0]) + (int(row[1])-2000)*12
            self.temperature = float(row[2])
            self.dewpoint = row[3]
            self.windspeed = row[4]
            self.soilwater_l1 = row[5]
            self.soilwater_l2 = row[6]
            self.soilwater_l3 = row[7]
            self.soilwater_l4 = row[8]
            self.precipitation = row[9]
    
    def get_date(self):
        return str(self.month % 12 + 1) + '/' + str(int(self.month/12)+2000)
