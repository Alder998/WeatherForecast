# Momentary file to run prediction

import PredictionService as p
from ReportingLibrary import Animations as ani
from ReportingLibrary import LocalizedWeather as locl

target = "temperature"
classModule = p.PredictionService(model='WeatherForecastModel_TimeSpaceSplit_' + target + '_393d_5Epochs',
                                  grid_step=0.22,
                                  start_date="2025-04-28", # Must be ALWAYS the day before the latest observation
                                  prediction_steps=48,
                                  predictiveVariables=['year','month','day','day_sin','hour','hour_sin','lat','lng'],
                                  variableToPredict=target)
predictions = classModule.NNPredict(confidence_levels=False, n_iter=None, loaded_scaler=None)

# Report Part
animation = ani.Animations().generateAnimationOnWeatherVariableFromDataFrame(dataFrame=predictions,
                                                                weatherVariable=target,
                                                                start_date=None,
                                                                end_date=None,
                                                                colorScale="rainbow",
                                                                save=False,
                                                                show=True)

timeSeriesForCity = locl.LocalizedWeather().getPredictionTimeSeriesOnTargetVariable (predictedDf = predictions,
                                                                                     city = 'Lavagna',
                                                                                     predictedVariable = target,
                                                                                     confidence_levels=False)