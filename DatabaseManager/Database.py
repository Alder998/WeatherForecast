# Database Class

from sqlalchemy import create_engine
import pandas as pd
import psycopg2

class Database:

    def __init__(self, database, user, password, host, port):
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        pass

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

    def appendDataToExistingTable (self,  dataFrame, existingTableName, drop_duplicates = True):

        # Before appending the table, be careful of not adding new columns, so check it
        # Load Existing
        existingData = self.getDataFromTable(existingTableName)
        # Check column Names: Check if the length of the columns is the same + all the dataFrame columns are present in
        # The existing DataFrame from SQL
        if (list(dataFrame.columns) == list(existingData.columns)):
            # Append
            newTable = self.saveTableInDatabase(dataFrame, existingTableName, mode='append')
            if drop_duplicates:
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
        data = self.getDataFromTable(tableName)
        print('Number of Rows: ' + '{:,}'.format(len(data[data.columns[0]])).replace(',', '.'))

        if len(columns) != 0:
            for col in columns:
                print('Number of Unique Values for column: ' + col + ':' +
                      ' {:,}'.format(len(data[col].unique())).replace(',', '.'))
        print('\n')
        return data

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












