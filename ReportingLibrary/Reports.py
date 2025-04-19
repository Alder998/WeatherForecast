# File to launch reports & colored Graphs
import LocalizedWeather as locl

#dataForPlot = locl.LocalizedWeather().getSingleWeatherTimeSeriesForCity(city = 'Chiavari',  # Available: any city in Italy > 9.000 inhabitants
#                                                                        aggregation = 'hourly',  # Available: 'hourly' | 'daily'
#                                                                        start_date='2025-01-01',
#                                                                        end_date='2025-01-20',
#                                                                        grid_step = 0.22)

comparison = locl.LocalizedWeather().compareWeatherForCities(city1 = 'Milano',
                                                             city2 = 'Lavagna',
                                                             aggregation = 'hourly',
                                                             start_date="2025-04-05",
                                                             end_date = "2025-04-14",
                                                             grid_step = 0.22
                                                             )