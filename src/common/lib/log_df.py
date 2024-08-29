import datetime
import os
import string
from typing import Any, Union
from matplotlib.axes import Axes
import pandas as pd

class LogDF(object):
    """Logger for logging dataframes into a csv file
    """
    def __init__(self, folder_path: string, filename_prefix:str='', index:Union[Axes, None]=None,
                 columns:Union[Axes,None]=None, should_write_index:bool=False):
        """Init a logger

        Args:
            folder_path (string): The path to the folder where the log should be saved into
            filename_prefix (str, optional): A prefix to the filename. Defaults to ''.
            index (Union[Axes, None], optional): The index for the dataframe. Defaults to None.
            columns (Union[Axes, None], optional): The columns for the dataframe. Defaults to None.
            should_write_index (bool, optional): Should save the index?. Defaults to False.
        """
        self.__path = os.path.join(folder_path, filename_prefix + datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f") + '.csv')
        self.__df = pd.DataFrame(index=index, columns=columns)
        self.__should_write_index = should_write_index
        
        # Create the file
        self.__save(self.__should_write_index, mode='w')
    
    @property
    def df(self):
        return self.__df
    
    @property
    def path(self):
        return self.__path
    
    def write(self, data:Union[Any, pd.DataFrame])->str:
        """Write data to file

        Args:
            data (Union[Any, pd.DataFrame]): The data to be written. If not a pd.Dataframe, it's converted to one

        Raises:
            f: Failing to convert the given data to a pd.DataFrame

        Returns:
            str: The path to the saved file
        """
        if type(data) is not pd.DataFrame:
            data = [data]
        
        try:
            self.__df = pd.DataFrame(data, columns=self.__df.columns)
        except Exception as ex:
            raise f"Can't convert 'data' to pd.DataFrame ({ex})"
            
        return self.__save(self.__should_write_index)
        
    def __save(self, index:bool=False, mode='a')->str:
        """Save the log to a csv file

        Args:
            index (bool, optional): Should the index be added to the file?. Defaults to False.
            mode (str, optional): Mode of writing: 'a' for pending or 'w' for overriding. Defaults to 'a'.

        Returns:
            str: The path to the saved file
        """
        self.__df.to_csv(self.__path, index=index, mode=mode, header=mode=='w')
        
        return self.__path