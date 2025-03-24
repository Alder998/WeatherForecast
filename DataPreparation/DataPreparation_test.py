import DataPreparation as dt
import numpy as np

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

for value in test_set:
    print(value.shape)

# Show shapes
print('Train Set Shapes:', classModule.getSetSize(train_set), '- Time Steps length:', len(train_set))
print('Train Labels Shapes:', classModule.getSetSize(train_labels), '- Time Steps length:', len(train_labels))
print('Test Set Shapes:', classModule.getSetSize(test_set), '- Time Steps length:', len(test_set))
print('Test Labels Shapes:', classModule.getSetSize(test_labels), '- Time Steps length:', len(test_labels))