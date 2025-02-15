import DataService as dt

start_date = "2025-01-01"
end_date = "2025-01-30"
grid_step = 0.22

dataExtract = dt.DataService().getWeatherDataForPointGrid(grid_step = grid_step,
                                            start_date = start_date,
                                            end_date = end_date, subset = 'all')

#print(dataExtract)