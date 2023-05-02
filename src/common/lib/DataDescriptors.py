import logging
import os
import traceback
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
#from src.common.lib.globals import CountsDF  - should not defined here - circular

#file_name='./data/raw/SpinningDisk\\./batch3\\WT\\panelA\\cond1\\rep1\\DAPI\\R11_w1confDAPI_s1.tif'  # look like chimera
# DAPI etc. should be taken from PreprocessingConfig  MARKERS_TO_INCLUDE
MARKERS_TO_INCLUDE = ['DAPI', 'GFP']
BATCH_Idenetifer = 'batch3'  # the 3 should be known before hand somehow
DATA_DESCRIPTORS_TAGS = ['BATCH','SNP','PANEL','CONDITION','REP','MARKER', 'TILE']  # maybe CELLLINE & not SNP


def Parse2Descriptors(file_name):
    try:
        line = ''
        name = os.path.normpath(file_name)
        parsed=name.split(os.sep)
        print(parsed)
        i2= len(parsed) # parsed.index(MARKERS_TO_INCLUDE)
        i1 =parsed.index(BATCH_Idenetifer)
        # I assume there is an internal order - to verify this

        for idx,data_item in zip((range(i1,(i2))),DATA_DESCRIPTORS_TAGS):
            line += data_item + ' : ' + str(parsed[idx]) + ', '
        return line

    except Exception as ex:
        logging.error(traceback.format_exc())
        print(ex.__context__)
        print('bad Parse2Descriptors')
        return line



class FeaturesHolder(object):

    def __init__(self):
        self.bFirstTime = True
        self.BATCH_Idenetifer = 'batch3'  # the 3 should be known before hand somehow
        self.DATA_DESCRIPTORS_TAGS = ['BATCH','SNP','PANEL','CONDITION','REP','MARKER', 'TILE']
        self.BATCH_Idenetifer = 'batch3'
        self.BATCH_Idenetifer_i = 0
        self.RowIdentifierName = 'Site\Image'
        self.DF = None
        self.ToPath = None   #IS check if you can not save & grab in need

    def AddLine(self, file_name, Vector):
        try:

            name = os.path.normpath(file_name)
            name = name.split(os.sep)
            if self.bFirstTime:
                self.BATCH_Idenetifer_i = name.index(BATCH_Idenetifer)
                cellscol_names = [("Cell"+str(i)) for i in range(len(Vector))]
                DF1 = DataFrame.from_dict({1:name[self.BATCH_Idenetifer_i:]}, orient = 'index', columns = [self.DATA_DESCRIPTORS_TAGS])
                DF2 = DataFrame.from_dict({1: list(Vector)}, orient = 'index', columns = [cellscol_names])
                self.DF =pd.merge(left = DF1, right = DF2, how = 'outer', left_index = True, right_index = True)
                self.bFirstTime = False
                return self
            i = len(self.DF)
            self.DF.loc[i+1] = name[self.BATCH_Idenetifer_i:] + list(Vector)
            return self

        except Exception as ex:
            logging.error(traceback.format_exc())
            print(ex.__context__)
            print('bad FeaturesHolder:AddLine')


    def Save(self):
        try:
            self.DF.to_csv(self.ToPath)
        except Exception as ex:
            #logging.error(traceback.format_exc())
            #print(ex.__context__)
            print('FeaturesHolder: No Save')

    def SetName(self, Path):
        self.ToPath = Path