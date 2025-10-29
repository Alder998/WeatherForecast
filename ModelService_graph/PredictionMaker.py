# File to run prediction
import json
import PredictionService as p
from ReportingLibrary import Animations as ani
from ReportingLibrary import LocalizedWeather as locl

model_name = "graph-3mo-1v-96h"

# Instantiate the class
prediction_data = p.PredictionService(model=model_name).predictWithStoredModel(grid_step=0.22, start_date="2025-10-07")

# Report Part
animation = ani.Animations().generateAnimationOnWeatherVariableFromDataFrame(dataFrame=prediction_data,
                                                                             weatherVariable="temperature",
                                                                             start_date=None,
                                                                             end_date=None,
                                                                             colorScale="rainbow",
                                                                             save=False,
                                                                             show=True)

timeSeriesForCity = locl.LocalizedWeather().getPredictionTimeSeriesOnTargetVariable (predictedDf=prediction_data,
                                                                                     city='Lavagna',
                                                                                     predictedVariable="temperature",
                                                                                     confidence_levels=False)