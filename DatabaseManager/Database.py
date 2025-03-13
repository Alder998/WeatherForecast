# Database Class
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
from DatabaseManager import DatabasePlugin_dask as dk
import dask.array as da

class Database:

    def __init__(self, database, user, password, host, port):
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        pass

    # Very basic method to execute a query
    def executeQuery (self, query):
        engine = create_engine('postgresql://' + self.user + ':' + self.password + '@' +
                               self.host + ':' + str(self.port) + '/' + self.database)
        query = query
        dataFromQuery = pd.read_sql(query, engine)

        return dataFromQuery

    # Get table Data from a Database Connection
    def getDataFromTable (self, tableName):

        engine = create_engine('postgresql://' + self.user + ':' + self.password + '@' +
                               self.host + ':' + str(self.port) + '/' + self.database)
        query = 'SELECT * FROM public."' + tableName + '"'
        dataFromQuery = pd.read_sql(query, engine)

        return dataFromQuery

    # Save a Table from DataFrame Format to an SQL Table
    def saveTableInDatabase (self, dataFrame, tableName, mode = 'create'):

        connection = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
        engine = create_engine('postgresql://' + self.user + ':' + self.password + '@' +
                               self.host + ':'+ str(self.port) + '/' + self.database)
        dataFrame.to_sql(tableName, engine, if_exists=mode, index=False)
        connection.close()

        return dataFrame

    # Method to get all the Tables in a given Database
    def getAllTablesInDatabase (self):

        # Get the engine
        engine = create_engine('postgresql://' + self.user + ':' + self.password + '@' +
                               self.host + ':'+ str(self.port) + '/' + self.database)
        query = ("SELECT table_schema, table_name" +
                 " FROM information_schema.tables \n" +
                 " WHERE table_type = 'BASE TABLE' AND table_schema NOT IN ('pg_catalog', 'information_schema') \n" +
                 " ORDER BY table_schema, table_name;")
        allTables = pd.read_sql(query, engine)

        return allTables['table_name']

    # Short Method to create a Table
    def createTable (self, dataFrame, tableName):
        # Get all the tables
        tablePresent = self.checkIfTableIsInDatabase(tableName)
        # Check if the table is in the Database
        if tablePresent:
            inputAnswer = input('You are currently overwriting an existing Table. If you want to continue with this action, please type "Y": ')
            if inputAnswer.lower() == 'y':
                action = self.saveTableInDatabase(dataFrame, tableName, mode='replace')
            else:
                raise Exception('A Table with the same name is already Existing!')
        else:
            action = self.saveTableInDatabase(dataFrame, tableName, mode='replace')
        return action

    def appendDataToExistingTable (self,  dataFrame, existingTableName, drop_duplicates = True, dask = False):

        # Before appending the table, be careful of not adding new columns, so check it
        # Load Existing
        if dask:
            existingData = dk.Database_dask(self.database, self.user, self.password, self.host,
                                            str(self.port)).getDataFromTable(existingTableName)
        else:
            existingData = self.getDataFromTable(existingTableName)

        # Update index
        dataFrame = self.updateIndex(existingTableName, dataFrame)

        # Check column Names: Check if the length of the columns is the same + all the dataFrame columns are present in
        # The existing DataFrame from SQL
        if (list(dataFrame.columns) == list(existingData.columns)):
            # Append
            newTable = self.saveTableInDatabase(dataFrame, existingTableName, mode='append')
            if drop_duplicates:
                if dask:
                    tableNoDuplicates = dk.Database_dask(database = self.database, user = self.user, password = self.password,
                                                         host = self.host, port = str(self.port)).getDataFromTable(existingTableName).drop_duplicates()
                else:
                    tableNoDuplicates = self.getDataFromTable(existingTableName).drop_duplicates()
                return tableNoDuplicates
            else:
                return newTable
        else:
            raise Exception('The table that you are trying to append has unmatched columns than the existing one!')

    # Method to have Table Statistics from dataset
    def getTableStatisticsFromQuery (self, tableName, columns = []):

        print('\n')
        print('-- TABLE: ' + tableName + ' --')
        # Get the data from query

        print('Number of Rows: ' + '{:,}'.format(self.executeQuery('SELECT MAX(row_number) FROM public."' +
                                                tableName + '"')['max'][0]).replace(',', '.'))

        if len(columns) != 0:
            for col in columns:
                print('Number of Unique Values for column: ' + col + ':' +
                      ' {:,}'.format(self.executeQuery('SELECT COUNT(DISTINCT '+ col +
                            ') FROM public."'+ tableName + '"')['count'][0]).replace(',', '.'))
        print('\n')

    def excludeValuesAlreadyPresentFromTable (self, dataFrame, tableName, keyColumn):

        # get Existing Table from Database
        existingData = self.getDataFromTable(tableName)
        # Exclude data from the current Database that are present in the existing Database
        data = dataFrame[~dataFrame[keyColumn].isin(existingData[keyColumn])].reset_index(drop=True)

        # Get some Logs
        print('-- Removal from DataFrame values present in the SQL table: ' + tableName + '--')
        print('Number of unique Keys before Removal: ' + '{:,}'.format(len(dataFrame[keyColumn].unique())).replace(',', '.'))
        print('Number of unique Keys after Removal: ' + '{:,}'.format(len(data[keyColumn].unique())).replace(',', '.'))

        return data

    def checkIfTableIsInDatabase (self, tableName):

        # get all the tables in Database
        allTables = self.getAllTablesInDatabase()
        # Case when there are no observations in the all Table Database
        if len(allTables) > 0:
            if pd.Series(tableName).isin(allTables)[0]:
                return True
            else:
                return False
        else:
            return False

    def updateIndex (self, originalTableName, dataFrame):

        # Create the index (required for Dask and for this function not to crash)
        # Get the latest index directly from query
        startIndex = self.executeQuery('SELECT MAX(row_number) FROM public."'+ originalTableName + '"')['max'][0]

        # Create the index
        rowNumberIndex = pd.DataFrame(range(startIndex + 1,
                        startIndex + 1 + len(dataFrame[dataFrame.columns[0]]))).set_axis(['row_number'],
                                                                                    axis = 1)
        # Concatenate the new index and the dataFrame
        dataFrame = pd.concat([rowNumberIndex, dataFrame], axis = 1)
        dataFrame = dataFrame.set_index("row_number")

        return dataFrame












