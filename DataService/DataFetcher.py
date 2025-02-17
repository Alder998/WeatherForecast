import DataService as dt
from ReportingLibrary import Animations as ani

start_date = "2025-01-01"
end_date = "2025-02-01"
grid_step = 0.22

dataExtract = dt.DataService().getWeatherDataForPointGrid(grid_step = grid_step,
                                            start_date = start_date,
                                            end_date = end_date, subset = 'all')

# Animation Library to see the geospatial data
# available: 'temperature' | 'precipitation' | 'windSpeed' | 'humidity_mean' | 'cloudCover' | 'pressure_msl'
animation = ani.Animations().generateAnimationOnWeatherVariable(grid_step=grid_step,
                                                                weatherVariable="windSpeed",
                                                                start_date=start_date,
                                                                end_date=end_date,
                                                                colorScale="rainbow",
                                                                save=False)

#print(dataExtract)