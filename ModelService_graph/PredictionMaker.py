# File to run prediction
import json
import PredictionService as p
from ReportingLibrary import Animations as ani
from ReportingLibrary import LocalizedWeather as locl

model_name = "graph-3mo-2v-96h"

# Instantiate the class
prediction_data = p.PredictionService(model=model_name).predictWithStoredModel(grid_step=0.22, start_date="2025-10-07")

# Report Part
try:
    animation = ani.Animations().generateAnimationOnWeatherVariableFromDataFrame(dataFrame=prediction_data,
                                                                                 weatherVariable="windSpeed",
                                                                                 start_date=None,
                                                                                 end_date=None,
                                                                                 colorScale="rainbow",
                                                                                 save=False,
                                                                                 show=True)
except:
    print("No internet available! Map Graph will not be shown!")

timeSeriesForCity = locl.LocalizedWeather().getPredictionTimeSeriesOnTargetVariable (predictedDf=prediction_data,
                                                                                     city='Trento',
                                                                                     predictedVariable="windSpeed",
                                                                                     confidence_levels=False)