# Momentary file to run prediction

import PredictionService as p
from ReportingLibrary import Animations as ani
from ReportingLibrary import LocalizedWeather as locl

classModule = p.PredictionService(model='WeatherForecastModel_TimeSpaceSplit_temperature_89d_15Epochs',
                                  grid_step=0.22,
                                  start_date="2025-03-31",
                                  prediction_steps=96)
predictions = classModule.NNPredict(confidence_levels=False, n_iter = None)

# Report Part
animation = ani.Animations().generateAnimationOnWeatherVariableFromDataFrame(dataFrame=predictions,
                                                                weatherVariable="temperature",
                                                                start_date=None,
                                                                end_date=None,
                                                                colorScale="rainbow",
                                                                save=False,
                                                                show=True)

timeSeriesForCity = locl.LocalizedWeather().getPredictionTimeSeriesOnTargetVariable (predictedDf = predictions,
                                                                                     city = 'Lavagna',
                                                                                     predictedVariable = 'temperature',
                                                                                     confidence_levels=False)