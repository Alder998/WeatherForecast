import DataPreparation as dt

# Instantiate the class
classModule = dt.DataPreparation()
# get the data creating a data subset (2-3 months)
data = classModule.getDataWindow(grid_step=0.22,
                                 start_date='2025-01-01',
                                 end_date='2025-01-15')
# Now, reshape
dataForModel = classModule.adaptDataForModel(dataFrame=data,
                                              predictiveVariables=['date', 'latitude', 'longitude'],
                                              variableToPredict='precipitation')
# Train-test split
t = classModule.timeAndSpaceSplit(dataForModel)

#print(dataForModel)