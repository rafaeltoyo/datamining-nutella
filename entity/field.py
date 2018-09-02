#======================================================================================================================#
#   class Field
#----------------------------------------------------------------------------------------------------------------------#
#   :author: Rafael Toyomoto
#======================================================================================================================#

from entity.history import History


class Component(object):

    def __init__(self, label=""):
        self.label = label
        self.sl_1 = 0
        self.sl_2 = 0
        self.sl_3 = 0
        self.sl_4 = 0
        self.sl_5 = 0
        self.sl_6 = 0
        self.sl_7 = 0

    @staticmethod
    def export(row, label=""):
        o = Component(label=label)
        o.sl_1 = int(row[label + "_sl1"])
        o.sl_2 = int(row[label + "_sl2"])
        o.sl_3 = int(row[label + "_sl3"])
        o.sl_4 = int(row[label + "_sl4"])
        o.sl_5 = int(row[label + "_sl5"])
        o.sl_6 = int(row[label + "_sl6"])
        o.sl_7 = int(row[label + "_sl7"])


class Field(object):

    def __init__(self, id):

        self.id = id

        self.bdricm = 0
        self.bdrlog = 0
        self.bdticm = 0

        self.bldfie = Component(label="BLDFIE")
        self.cecsol = Component(label="CECSOL")
        self.clyppt = Component(label="CLYPPT")
        self.crfvol = Component(label="CRFVOL")
        self.ocstha = Component(label="OCSTHA")
        self.orcdrc = Component(label="ORCDRC")
        self.phihox = Component(label="PHIHOX")
        self.phikcl = Component(label="PHIKCL")
        self.sltppt = Component(label="SLTPPT")
        self.sndppt = Component(label="SNDPPT")

        self.history = {}

    def add_history(self, history):
        self.history[str(history.year) + str(history.month)] = history
        return self

    @staticmethod
    def export(row):

        o = Field(int(row["field"]))

        o.bdricm = int(row["BDRICM_BDRICM_M"])
        o.bdrlog = int(row["BDRLOG_BDRLOG_M"])
        o.bdticm = int(row["BDTICM_BDTICM_M"])

        o.bldfie = Component.export(row, label="BLDFIE")
        o.cecsol = Component.export(row, label="CECSOL")
        o.clyppt = Component.export(row, label="CLYPPT")
        o.crfvol = Component.export(row, label="CRFVOL")
        o.ocstha = Component.export(row, label="OCSTHA")
        o.orcdrc = Component.export(row, label="ORCDRC")
        o.phihox = Component.export(row, label="PHIHOX")
        o.phikcl = Component.export(row, label="PHIKCL")
        o.sltppt = Component.export(row, label="SLTPPT")
        o.sndppt = Component.export(row, label="SNDPPT")
