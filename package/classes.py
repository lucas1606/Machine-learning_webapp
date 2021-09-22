from __future__ import annotations
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class Helpers:

    """
     Class with some property methods that suports the PreProcessor class 
    """

    def __repr__(self):
            return repr(self.df)
            
    @property
    def columns_names(self)->list:

        """
        Return --> list with all columns names
        """

        return list(self.df.columns)

    @property
    def columns_types(self)->list:
        types = []
        for column in list (self.df.dtypes):
            types.append(column.name)

        """
        Return --> list with the columns types
        """
        return types

    @property
    def na_lines(self)->list: 

        """
        Return --> list of all lines containing Na values
        """
        return list(self.df.loc[self.df.isna().sum(axis=1)>0].index)

    @property
    def na_columns(self):

        """
        Return --> list of a all columns containging Na values
        """
        return list(self.df.loc[:,self.df.isna().sum() > 0].columns)

class PreProcessor(Helpers): #Method is initiated 
    def __init__(self, df:DataFrame)->None:

        """
        Class is initiated with a Pandas DataFrame Object
        """
        self.df = df

    def remove_na_lines(self, max_na_proportion:float=None):

        """
        Method removes lines containg Na values acording to the criteria os max_na_proportion.
        max_na_proportion values from [0.0 to 1.0] --> Defines the max proportion of Na values a line can contain
         so it is kept in the Dataframe.
        If Value == None drops all the lines containg Na values
        """

        if max_na_proportion:
            drop_lines =  list(self.df.loc[self.df.isna().sum(axis=1)>len(self.df.columns)*max_na_proportion].index) 
        else:
            drop_lines = self.na_lines
        print(f'Lines removed:{drop_lines[0:4]}...')
        self.df = self.df.drop(drop_lines)
        return self.df

    def remove_na_columns(self, max_na_proportion=None):

        """
        Method removes columns containg Na values acording to the criteria os max_na_proportion.
        max_na_proportion values from [0.0 to 1.0] --> Defines the max proportion of Na values a columns can contain
        so it is kept in the Dataframe.

        If Value == None drops all the columns containg Na values
        """

        if max_na_proportion:
            drop_columns = list(self.df.loc[:,self.df.isnull().sum() > len(self.df.index)*max_na_proportion].columns)
        print(f'Columns removed{drop_columns[0:4]}...')
        self.df = self.df.drop(columns=drop_columns)
        return self.df

    def replace_na_values(self):

        """
        Method replaces all Na values according to it's columns dtype.
        If column.dtype == 'object' --> replaced_value == column.mode()
        If column.dtype == 'int64' --> replaced_value == column.meadian()
        if column.dtype == 'float64' --> replaced_value == column.median()
        """

        for column in self.na_columns:
            if self.df[column].dtype == 'object':
                self.df[column].fillna(self.df[column].mode(), inplace = True) 

            if self.df[column].dtype == 'int64':
                self.df[column].fillna(self.df[column].median(), inplace = True)

            if self.df[column].dtype == 'float64':
                self.df[column].fillna(self.df[column].median(), inplace = True)
        return self.df

    def encode_categorical(self):
        """
        Method encodes the categorical columns with are supose to be the ones whose the dtype == 'object'
        using the class Sklearn.preprocessing.LabelEncoder() 
        """
        label_maker = LabelEncoder()
        columns = list(self.df.loc[:,(self.df.dtypes == 'object')].columns)
        for col in columns:
            self.df[col] = label_maker.fit_transform(self.df[col].astype('str'))
        return self.df

    
    def normalize(self, norm_method:str=None):
        """   
        Method normalizes data using os np.sqrt() or np.log() functions.

        Parameters
        ----------
        norm_method:str : {'log', 'sqrt'}, default ='log'
            Define the normalization method that will be applyed to the DataFrame

        """
        if norm_method == 'log' or norm_method == None:
            for column in self.df.columns:
                if 0 in self.df[column].values:
                    continue
                else:
                    self.df.loc[:,[column]] = np.log(self.df[column])

        if norm_method == 'sqrt':
            for column in self.df.columns:
                if 0 > self.df[column].min():
                    pass
                else:
                    self.df[column] = np.sqrt(self.df[column])
        return self.df

    def feature_selection(self, objective_column:str=None, min_correlation:float=None):   
        """
        Method selects the best columns or features of the DataFrame according to their correlation 
        with the target variable
        
        Parameters
        ----------
        min_correlation:float64 : {0, 1.0}, default = 0.4
            Define the minimum correlation the column should have with the target variable so it is not erased
            from the DataFrame,the default value is 0.4 aan it representes a minnimum correlation of 40%

        objective_columns:str :['<Target_column_name>'], default = None
            Define the name of the target variable. The default target variable will be considered the last column 
            in the DataFrame
        """
        if objective_column == None:
            objective_column = list(self.df.iloc[:,-1:])[0]
        if min_correlation == None:
            min_correlation = 0.4

        corr_ranking =  self.df.corr()[objective_column].abs().sort_values(ascending = False)
        self.df = self.df.loc[:,corr_ranking >= min_correlation]
        
        return self.df
         
        
        





       


    
       
        

        







