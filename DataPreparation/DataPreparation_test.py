import DataPreparation as dt

# Instantiate the class
classModule = dt.DataPreparation()
# get the data creating a data subset (2-3 months)
data = classModule.getDataSubset(grid_step=0.22,
                                 start_date='2025-01-01',
                                 end_date='2025-01-05')
# Now, reshape
dataForModel = classModule.adaptDataForModel (dataFrame=data,
                                              predictiveVariables=['date', 'latitude', 'longitude'],
                                              variableToPredict='precipitation')
print(dataForModel)