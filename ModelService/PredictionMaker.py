# File to run prediction
import json
import PredictionService as p
from ReportingLibrary import Animations as ani
from ReportingLibrary import LocalizedWeather as locl

model_name = 'small-bidirectional-LSMT-uniform_time'

# Read the modelInfo
with open("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + model_name + '\\modelInfo.json', "r") as f:
    model_info = json.load(f)

# Instantiate the class
classModule = p.PredictionService(model="D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + model_name + "\\" + model_name + ".h5",
                                  grid_step=model_info["grid_step"],
                                  start_date="2025-07-09", # Must be ALWAYS the day before the latest observation
                                  prediction_steps=96,
                                  predictiveVariables=model_info["predictive_variables"],
                                  variableToPredict=model_info["target_variable"],
                                  timeVariables=model_info["time_variables"])
# Execute the prediction
predictions = classModule.NNPredict(confidence_levels=False, n_iter=None, loaded_scaler=None)

# Report Part
try:
    animation = ani.Animations().generateAnimationOnWeatherVariableFromDataFrame(dataFrame=predictions,
                                                                    weatherVariable=model_info["target_variable"].replace("_residual", ""),
                                                                    start_date=None,
                                                                    end_date=None,
                                                                    colorScale="rainbow",
                                                                    save=False,
                                                                    show=True)
except:
    print("No Connection for the map report! Passing to the following Report...")

timeSeriesForCity = locl.LocalizedWeather().getPredictionTimeSeriesOnTargetVariable (predictedDf = predictions,
                                                                                     city = 'Lavagna',
                                                                                     predictedVariable=model_info["target_variable"].replace("_residual", ""),
                                                                                     confidence_levels=False)