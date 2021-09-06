import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



class Helpers:
    '''
    Recives a csv relative path and extract ist propeties in its inicial estate
    Parameters
    '''
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

class NaSolver(Helpers): #Only basic functions
    def __init__(self, df:object):
        self.df = df

    def remove_na_lines(self, max_na_proportion=None):
        if max_na_proportion:
            drop_lines = list(self.df.loc[self.df.iloc[self.na_lines].isna().sum(axis=1) > len(self.df.columns)*max_na_proportion].index)
        else:
            drop_lines = self.na_lines
        print(f'Lines removed:{drop_lines[0:4]}...')
        self.df = self.df.drop(drop_lines)
        return self.df

    def remove_na_columns(self, max_na_proportion=None):
        if max_na_proportion:
            drop_columns = list(self.df.loc[:,self.df.isnull().sum() > len(self.df.index)*max_na_proportion].columns)
        print(f'Columns removed{drop_columns[0:4]}...')
        self.df = self.df.drop(columns=drop_columns)
        return self.df

    def replace_na_values(self):
        for column in self.na_columns:
            if self.df[column].dtype == 'object':
                self.df[column].fillna(self.df[column].mode(), inplace = True) 

            if self.df[column].dtype == 'int64':
                self.df[column].fillna(self.df[column].median(), inplace = True)

            if self.df[column].dtype == 'float64':
                self.df[column].fillna(self.df[column].median(), inplace = True)
        return self.df

    def encode_categorical(self):
        label_maker = LabelEncoder()
        columns = list(self.df.loc[:,(self.df.dtypes == 'object')].columns)
        for col in columns:
            self.df[col] = label_maker.fit_transform(self.df[col].astype('str'))
        return self.df




       


    
       
        

        







