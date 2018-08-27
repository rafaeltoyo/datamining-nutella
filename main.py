#==============================================================================#
#==============================================================================#

from entity.field import Field
from utils.loader import Loader

menor_data = None
for row in Loader(path="dataset/").load_fields():
    field = Field(row)

