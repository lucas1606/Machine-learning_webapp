import pandas as pd
import numpy as np
import sys

class Data:
    '''
    Recives a csv relative path and extract ist propeties in its inicial estate
    Parameters
    '''
    def __init__(self, path, sep=None):
        self._path = path
        self._sep = ','
        self._df = pd.read_csv(self._path, sep=self._sep)
        
    @property
    def df(self):
        return self._df

    def __repr__(self):
            return repr(self.df)
            
    @property
    def columns_names(self):
        return list(self.df.columns)

    @property
    def columns_types(self):
        types = []
        for column in list (self.df.dtypes):
            types.append(column.name)
        return types

    @property
    def na_lines(self): 
        return list(self.df.loc[self.df.isna().sum(axis=1)>0].index)

    @property
    def na_columns(self):
        return list(self.df.loc[:,self.df.isna().sum() > 0].columns)

class NaSolver(Data):
    def __init__(self, df:object):
        self._df = df
    def remove_na_lines(self, na_proportion=None):
        print(f'Lines removed:{self.na_lines}')
        self._df = self.df.drop(self.na_lines)
        return self._df


    
    
       
        

        


#sys.path.append(".")

