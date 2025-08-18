import json
import os
import sys
import PredictionService as p

# Add all folders for batch execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

model_name = 'small-bidirectional-LSMT-uniform_time'

# Read the modelInfo
with open("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + model_name + '\\modelInfo.json', "r") as f:
    model_info = json.load(f)

classModule = p.PredictionService(model="D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + model_name + "\\" + model_name + ".h5",
                                  grid_step=model_info["grid_step"],
                                  start_date="2025-07-31", # Must be ALWAYS the day before the latest observation
                                  prediction_steps=96,
                                  predictiveVariables=model_info["predictive_variables"],
                                  variableToPredict=model_info["target_variable"],
                                  timeVariables=model_info["time_variables"],
                                  prophet_params = {
                                      "re-train": True,
                                      "dataset_depth": 30,
                                      "prediction_steps": 100,
                                      "rolling_window_trend": 300,
                                  }).createProphetPrediction()
