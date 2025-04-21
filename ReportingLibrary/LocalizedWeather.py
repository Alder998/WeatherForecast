# Library for localized reports
from DatabaseManager import Database as db
from dotenv import load_dotenv
import os
from geopy.distance import geodesic
import pandas as pd
import matplotlib.pyplot as plt
from DatabaseManager import DatabasePlugin_dask as dk

class LocalizedWeather:
    def __init__(self):
        pass

    def databaseModule (self, dask=False):

        env_path = r"D:\PythonProjects-Storage\WeatherForecast\App_core\app.env"
        load_dotenv(env_path)
        database = os.getenv("database")
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host")
        port = os.getenv("port")

        # Instantiate the database Object, according whether it is Dask or Not
        dataClass = db.Database(database, user, password, host, port)

        if dask:
            dataClass = dk.Database_dask(database, user, password, host, port)
        else:
            dataClass = db.Database(database, user, password, host, port)

        return dataClass

    def findCityCoordinates (self, weatherData, city, start_date = None, end_date = None, aggregation = 'hourly'):

        # Data Class Instantiation (use dask only for the Weather Dataset that is massive, while for the cities one keep
        # the old method)
        dataClass = self.databaseModule()

        # Get the cities data for the specific city
        cities = dataClass.getDataFromTable("Cities_Italy")
        selectedCity = cities[['lat', 'lng']][cities['city'] == city].reset_index(drop=True).set_axis(['latitude', 'longitude'], axis = 1).reset_index(drop=True)
        if len(selectedCity['latitude']) == 0:
            raise Exception ("The city: " + city + " has NOT been found in the Database!")

        # Find the nearest Points with the greediest method possible
        # First, isolate the 716 grid points
        uniqueCoords = weatherData.drop_duplicates(subset=['latitude', 'longitude'])

        # Now, compute distance with the appropriate library
        target = (selectedCity['latitude'].values, selectedCity['longitude'].values)
        uniqueCoords['Distance_km'] = uniqueCoords.apply(
            lambda row: geodesic((row['latitude'], row['longitude']), target).kilometers, axis=1)

        # Log to see how far a weather Data Point has been found from the selected city
        print('Weather data point found: ' + str(round(uniqueCoords['Distance_km'].min(), 2)) + ' km from ' + city)

        minDistanceCoord = uniqueCoords[['latitude', 'longitude']][
            uniqueCoords['Distance_km'] == uniqueCoords['Distance_km'].min()].reset_index(drop=True)

        filteredWeatherData = weatherData[(weatherData['latitude'] == minDistanceCoord['latitude'].values[0]) &
                                          (weatherData['longitude'] == minDistanceCoord['longitude'].values[
                                              0])].reset_index(drop=True)
        # Aggregate if necessary
        if aggregation == 'daily':
            filteredWeatherData['date'] = pd.to_datetime(filteredWeatherData['date']).dt.date
            filteredWeatherData = filteredWeatherData.groupby('date', as_index=False).mean()

        # filter for date if requested
        if (start_date != None) & (end_date != None):
            dt_date_start = pd.to_datetime(start_date, format="%Y-%m-%d").date()
            dt_date_end = pd.to_datetime(end_date, format="%Y-%m-%d").date()
            filteredWeatherData = filteredWeatherData[
                (pd.to_datetime(filteredWeatherData['date']).dt.date > dt_date_start) &
                (pd.to_datetime(filteredWeatherData['date']).dt.date < dt_date_end)].reset_index(drop=True)
        filteredWeatherData['date'] = pd.to_datetime(filteredWeatherData['date'])

        return filteredWeatherData, uniqueCoords

    def getFilteredDatasetForCity (self, city, start_date = None, end_date = None, aggregation = 'hourly', grid_step = 0.22):

        # Data Class Instantiation (use dask only for the Weather Dataset that is massive, while for the cities one keep
        # the old method)
        dataClass = self.databaseModule()

        # Only narrow date filters are admitted
        weatherData = dataClass.executeQuery('SELECT * FROM public."WeatherForRegion_' + str(grid_step) +
                                             '" WHERE date BETWEEN ' + "'" + start_date + "'" + ' AND ' + "'" +
                                             end_date + "'")

        # Find the nearest Points with the greediest method possible
        filteredWeatherData, uniqueCoords = self.findCityCoordinates(weatherData = weatherData,
                                            city=city, start_date = start_date, end_date = end_date,
                                                                     aggregation = aggregation)

        return filteredWeatherData, uniqueCoords

    def getSingleWeatherTimeSeriesForCity (self, city, start_date = None, end_date = None,
                                           aggregation ='hourly', grid_step = 0.22):

        filteredWeatherData, distances = self.getFilteredDatasetForCity(city, start_date, end_date, aggregation, grid_step)

        # Now, plot the data we have found for the city
        filteredWeatherData = filteredWeatherData.sort_values(by='date', ascending=True)
        filteredWeatherData = filteredWeatherData.set_index(pd.to_datetime(filteredWeatherData['date']))

        fig, axes = plt.subplots(3, 2, figsize=(15, 8))

        # Temperature
        axes[0, 0].plot(filteredWeatherData['temperature'], label='temperature (°C)', color='blue')
        axes[0, 0].set_title('Temperature Evolution')
        axes[0, 0].legend()

        # Precipitation
        axes[0, 1].plot(filteredWeatherData['precipitation'], label='precipitation (mm)', color='forestgreen')
        axes[0, 1].set_title('Precipitation Evolution')
        axes[0, 1].legend()

        # Humidity
        axes[1, 0].plot(filteredWeatherData['humidity_mean'], label='humidity level', color='purple')
        axes[1, 0].set_title('Humidity Evolution')
        axes[1, 0].legend()

        # Wind Speed
        axes[1, 1].plot(filteredWeatherData['windSpeed'], label='Wind Speed', color='red')
        axes[1, 1].set_title('Wind Speed')
        axes[1, 1].legend()

        # Cloud coverage
        axes[2, 0].plot(filteredWeatherData['cloudCover'], label='Cloud Coverage (%)', color='grey')
        axes[2, 0].set_title('Cloud Coverage')
        axes[2, 0].legend()

        # Pressure level
        axes[2, 1].plot(filteredWeatherData['pressure_msl'], label='Pressure', color='orange')
        axes[2, 1].set_title('Pressure')
        axes[2, 1].legend()

        fig.suptitle("(" + str((pd.to_datetime(filteredWeatherData['date']).dt.date).min()) +
                     " - " + str((pd.to_datetime(filteredWeatherData['date']).dt.date).max()) +  ") " + f"Weather Data for the city: {city} ({round(distances['Distance_km'].min(), 2)} km from {city})",
            fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.show()

    # Plotting Function to compare weather of two different cities

    def compareWeatherForCities(self, city1, city2, start_date, end_date, aggregation = 'hourly', grid_step = 0.22):

        city1Filter, distances1 = self.getFilteredDatasetForCity(city1, start_date, end_date, aggregation, grid_step)
        city2Filter, distances2 = self.getFilteredDatasetForCity(city2, start_date, end_date, aggregation, grid_step)

        # Now, Plot the comparison
        city1Filter = city1Filter.sort_values(by='date', ascending=True)
        city1Filter = city1Filter.set_index(pd.to_datetime(city1Filter['date']))

        city2Filter = city2Filter.sort_values(by='date', ascending=True)
        city2Filter = city2Filter.set_index(pd.to_datetime(city2Filter['date']))

        fig, axes = plt.subplots(3, 2, figsize=(15, 8))

        # Temperature
        axes[0, 0].plot(city1Filter['temperature'], label=city1, color='blue')
        axes[0, 0].plot(city2Filter['temperature'], label=city2, color='skyblue', linestyle = 'dashed')
        axes[0, 0].set_title('Temperature Evolution')
        axes[0, 0].legend()

        # Precipitation
        axes[0, 1].plot(city1Filter['precipitation'], label=city1, color='forestgreen')
        axes[0, 1].plot(city2Filter['precipitation'], label=city2, color='limegreen',  linestyle = 'dashed')
        axes[0, 1].set_title('Precipitation Evolution')
        axes[0, 1].legend()

        # Humidity
        axes[1, 0].plot(city1Filter['humidity_mean'], label=city1, color='purple')
        axes[1, 0].plot(city2Filter['humidity_mean'], label=city2, color='violet', linestyle='dashed')
        axes[1, 0].set_title('Humidity Evolution')
        axes[1, 0].legend()

        # Wind Speed
        axes[1, 1].plot(city1Filter['windSpeed'], label=city1, color='red')
        axes[1, 1].plot(city2Filter['windSpeed'], label=city2, color='lightcoral', linestyle = 'dashed')
        axes[1, 1].set_title('Wind Speed')
        axes[1, 1].legend()

        # Cloud coverage
        axes[2, 0].plot(city1Filter['cloudCover'], label=city1, color='dimgray')
        axes[2, 0].plot(city2Filter['cloudCover'], label=city2, color='grey', linestyle = 'dashed')
        axes[2, 0].set_title('Cloud Coverage')
        axes[2, 0].legend()

        # Pressure level
        axes[2, 1].plot(city1Filter['pressure_msl'], label=city1, color='orange')
        axes[2, 1].plot(city2Filter['pressure_msl'], label=city2, color='gold', linestyle = 'dashed')
        axes[2, 1].set_title('Pressure')
        axes[2, 1].legend()

        fig.suptitle("(" + str((pd.to_datetime(city2Filter['date']).dt.date).min()) +
                     " - " + str((pd.to_datetime(city2Filter['date']).dt.date).max()) +  ") " +
                     f"Weather Comparison: {city1} ({round(distances1['Distance_km'].min(), 2)} km from {city1}) VS {city2} ({round(distances2['Distance_km'].min(), 2)} km from {city2})",
            fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.show()

    # Method to have weather Prediction plot from DataFrame
    def getPredictionTimeSeriesOnTargetVariable (self, predictedDf, city, predictedVariable, confidence_levels = False):

        dataClass = self.databaseModule()
        # isolate the city Name
        # Get the cities data for the specific city
        cities = dataClass.getDataFromTable("Cities_Italy")
        selectedCity = cities[['lat', 'lng']][cities['city'] == city].reset_index(drop=True).set_axis(['latitude', 'longitude'], axis = 1).reset_index(drop=True)
        if len(selectedCity['latitude']) == 0:
            raise Exception ("The city: " + city + " has NOT been found in the Database!")

        cityCoord, coords = self.findCityCoordinates(weatherData=predictedDf, city=city,
                                 start_date=None, end_date=None, aggregation='hourly')

        # Gest the last n-observations from data
        cityCoord_real, coords_real = self.getLastObservationsFromCityAndVariable (grid_step=0.22, city=city)

        print(cityCoord['date'].min())
        print(cityCoord_real['date'].max())

        # concatenate with the predictions
        allCityCoords = pd.concat([cityCoord_real[['date', predictedVariable]], cityCoord[['date', predictedVariable]]], axis = 0).reset_index(drop=True)
        allCityCoords = allCityCoords.sort_values(by='date', ascending=True).reset_index(drop=True)

        # A little bit of logging
        print(predictedVariable + ' Min registered from prediction: ' + str(cityCoord[predictedVariable].min()) + ' on ' +
              pd.to_datetime(cityCoord['date'][cityCoord[predictedVariable] == cityCoord[predictedVariable].min()]).dt.strftime("%d/%m/%Y %H:%M").reset_index(drop = True)[0])
        print(predictedVariable + ' Max registered from prediction: ' + str(cityCoord[predictedVariable].max()) + ' on ' +
              pd.to_datetime(cityCoord['date'][cityCoord[predictedVariable] == cityCoord[predictedVariable].max()]).dt.strftime("%d/%m/%Y %H:%M").reset_index(drop = True)[0])
        print('Average ' + predictedVariable + ': ' + str(cityCoord[predictedVariable].mean()))

        # Now, plot
        plt.figure(figsize = (12, 4))
        # Plot Actual
        plt.plot(allCityCoords[allCityCoords['date'] < cityCoord_real['date'].max()].set_index('date')[predictedVariable],
                 color = 'blue', label = 'Actual data')
        # plot Prediction with different color
        plt.plot(allCityCoords[allCityCoords['date'] > cityCoord_real['date'].max()].set_index('date')[predictedVariable],
                 color = 'red',linestyle = 'dashed', label = 'Prediction')
        plt.axvline(cityCoord_real['date'].max(), color='black', linestyle='dashed')
        if confidence_levels:
            plt.fill_between(allCityCoords['date'], allCityCoords.set_index('date')[predictedVariable + '_lower'],
                             allCityCoords.set_index('date')[predictedVariable + '_upper'], color = 'pink')
        plt.title(city + ' - ' + predictedVariable + ' prediction Evolution')
        plt.xlabel('Date')
        plt.ylabel(predictedVariable)
        plt.legend()
        plt.show()

        return cityCoord, coords

    # function to get the latest observation on a city
    def getLastObservationsFromCityAndVariable (self, grid_step, city, steps_behind = 10):

        dataClass = self.databaseModule()
        lastDate = dataClass.executeQuery('SELECT MAX(date) FROM public."WeatherForRegion_' + str(grid_step) + '"')['max'][0]

        lastDate = pd.to_datetime(str(lastDate))
        lastDate_nDaysBefore = lastDate - pd.Timedelta(days=steps_behind)

        # Now, isolate the last n-dates
        cityCoord, coords = self.getFilteredDatasetForCity(city, start_date=lastDate_nDaysBefore.strftime("%Y-%m-%d"),
                                                             end_date=lastDate.strftime("%Y-%m-%d"),
                                                             aggregation='hourly', grid_step=0.22)

        return cityCoord, coords













