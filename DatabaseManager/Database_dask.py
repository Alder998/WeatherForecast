# Class to implement the Database reader into dask framework

import dask.dataframe as dd
from sqlalchemy import create_engine
import pandas as pd
import psycopg2

class Database_dask:

    def __init__(self, database, user, password, host, port):
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        pass

    def getDataFromTable (self, tableName):

        # intantiate the engine (via string)
        engine_str = 'postgresql://' + self.user + ':' + self.password + '@' + self.host + ':' + str(self.port) + '/' + self.database
        # Get the database
        df = dd.read_sql_table(tableName, con=engine_str, index_col="row_number", npartitions=10)
        return df
