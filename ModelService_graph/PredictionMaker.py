# File to run prediction
import json
import PredictionService as p
from ReportingLibrary import Animations as ani
from ReportingLibrary import LocalizedWeather as locl

model_name = "graph-3mo-1v-96h"

# Instantiate the class
classModule = p.PredictionService(model=model_name).prepareDataForModel()


# Report Part
#try:
#    animation = ani.Animations().generateAnimationOnWeatherVariableFromDataFrame(dataFrame=predictions,
#                                                                    weatherVariable=model_info["target_variable"].replace("_residual", ""),
#                                                                    start_date=None,
#                                                                    end_date=None,
#                                                                    colorScale="rainbow",
#                                                                    save=False,
#                                                                    show=True)
#except:
#    print("No Connection for the map report! Passing to the following Report...")
#
#timeSeriesForCity = locl.LocalizedWeather().getPredictionTimeSeriesOnTargetVariable (predictedDf = predictions,
#                                                                                     city = 'Milano',
#                                                                                     predictedVariable=model_info["target_variable"].replace("_residual", ""),
#                                                                                     confidence_levels=False)