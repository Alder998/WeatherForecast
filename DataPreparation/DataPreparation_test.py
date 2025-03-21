import DataPreparation as dt

# Instantiate the class
classModule = dt.DataPreparation(grid_step=0.22)
# get the data creating a data subset (2-3 months)
data = classModule.getDataWindow(start_date='2025-01-01',
                                 end_date='2025-01-10')
# Train-test split
train_set, test_set, train_labels, test_labels = classModule.timeAndSpaceSplit(dataset=data,
                                                    test_size=0.20,
                                                    predictiveVariables=['date', 'latitude', 'longitude'],
                                                    variableToPredict='precipitation')

#print('Train Set Shape:', train_labels.shape)
#print('Test Labels Shape:', test_labels.shape)

print('Train set:', len(train_set))