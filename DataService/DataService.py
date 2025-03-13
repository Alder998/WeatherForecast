# Weather Data-Fetcher class

import requests
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from DatabaseManager import Database as db
from dotenv import load_dotenv
import os

class DataService:
    def __init__(self):
        pass

    # Utils-like Method to get the Database .env
    def databaseModule (self):

        env_path = r"D:\PythonProjects-Storage\WeatherForecast\App_core\app.env"
        load_dotenv(env_path)
        database = os.getenv("database")
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host")
        port = os.getenv("port")

        # Instantiate the database Object
        dataClass = db.Database(database, user, password, host, port)

        return dataClass

    def getWheatherByParams(self, latitude, longitude, start_date, end_date, save=False):

        # OpenMeteo URL to get Historical weather data
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,precipitation,relative_humidity_2m,windspeed_10m,cloudcover,pressure_msl&timezone=Europe/Rome"

        # Esegui la richiesta
        response = requests.get(url)
        data = response.json()

        weatherDBExtract = []
        #print('Getting Weather Data for latitude: ' + str(round(latitude, 2)) + ' and longitude: ' + str(round(longitude, 2)))
        for i, date in enumerate(data["hourly"]["time"]):
            # Log
            temperature = data["hourly"]["temperature_2m"][i]
            precipitation = data["hourly"]["precipitation"][i]
            humidity = data["hourly"]["relative_humidity_2m"][i]
            wind = data["hourly"]["windspeed_10m"][i]
            cloudCover = data["hourly"]["cloudcover"][i]
            pressure = data["hourly"]["pressure_msl"][i]

            # Structure the data in a Database
            singleDatabase = pd.concat([pd.DataFrame(pd.Series(date)).set_axis(['date'], axis=1),
                                        pd.DataFrame(pd.Series(latitude)).set_axis(['latitude'], axis=1),
                                        pd.DataFrame(pd.Series(longitude)).set_axis(['longitude'], axis=1),
                                        pd.DataFrame(pd.Series(temperature)).set_axis(['temperature'], axis=1),
                                        pd.DataFrame(pd.Series(precipitation)).set_axis(['precipitation'], axis=1),
                                        pd.DataFrame(pd.Series(humidity)).set_axis(['humidity_mean'], axis=1),
                                        pd.DataFrame(pd.Series(wind)).set_axis(['windSpeed'], axis=1),
                                        pd.DataFrame(pd.Series(cloudCover)).set_axis(['cloudCover'], axis=1),
                                        pd.DataFrame(pd.Series(pressure)).set_axis(['pressure_msl'], axis=1)],
                                       axis=1)
            weatherDBExtract.append(singleDatabase)
        weatherDBExtract = pd.concat([df for df in weatherDBExtract], axis=0).reset_index(drop=True)

        return weatherDBExtract

    def createPointsGrid(self, grid_step, plot=False):

        # Load the Map
        world = gpd.read_file(
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson")
        italy = world[world["NAME"] == "Italy"]

        # Get the Bounders Points
        minx, miny, maxx, maxy = italy.total_bounds

        # Get the grid according to the granularity
        lon_points = np.arange(minx, maxx, grid_step)
        lat_points = np.arange(miny, maxy, grid_step)
        grid = np.array(np.meshgrid(lon_points, lat_points)).T.reshape(-1, 2)

        # Convert to a GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid[:, 0], grid[:, 1]), crs="EPSG:4326")

        # Filter for the Inside-Region Points
        grid_inside_region = grid_gdf[grid_gdf.within(italy.geometry.iloc[0])].reset_index(drop=True)
        print('Eligible Points: ' + str(len(grid_inside_region['geometry'])))

        # Transform into DataFrame
        coordinates = pd.concat([pd.Series(grid_inside_region.geometry.x),
                                 pd.Series(grid_inside_region.geometry.y)], axis=1).set_axis(
                            ['lng', 'lat'], axis=1)

        # Instantiate a Database Object
        dataClass = self.databaseModule()
        # Save into the Database
        tableName = 'gridPoints_' + str(grid_step)
        tablePresent = dataClass.checkIfTableIsInDatabase(tableName)
        if not tablePresent:
            dataClass.createTable(coordinates, tableName)

        if plot:
            # Plot the data
            fig, ax = plt.subplots(figsize=(8, 10))
            italy.plot(ax=ax, color='white', edgecolor='black')
            grid_inside_region.plot(ax=ax, color='red', markersize=5, alpha=0.6)

            plt.title("Points in Region", fontsize=14)
            plt.xticks([])
            plt.yticks([])
            plt.show()

        return grid_inside_region

    def getWeatherDataForPointGrid (self, grid_step, start_date, end_date, subset = 'all', dask=False):

        points = self.createPointsGrid(grid_step, plot=False)
        coordinates = pd.concat([pd.Series(points.geometry.x), pd.Series(points.geometry.y)], axis=1).set_axis(
            ['lng', 'lat'], axis=1)
        if subset != 'all':
            coordinates = coordinates[0:subset]

        wForCoord = []
        if not coordinates.empty:
            try:
                for i in range(len(coordinates['lat'])):
                    print('Loading Wheather Data... ' + str(round((i / len(coordinates['lat'])) * 100, 2)) + '%')
                    latitude = coordinates['lat'][i]
                    longitude = coordinates['lng'][i]
                    wData = self.getWheatherByParams(latitude, longitude, start_date, end_date)
                    wForCoord.append(wData)
                wForCoord = pd.concat([df for df in wForCoord], axis=0).reset_index(drop=True)
            except:
                raise Exception ("Max Retries or Connection lost! Impossible to carry out each point download!")
                #wForCoord = pd.concat([df for df in wForCoord], axis=0).reset_index(drop=True)

            # Load Database
            # Table Name wrt granularity of abservations
            tableName = 'WeatherForRegion_' + str(grid_step)
            tablePresent = self.databaseModule().checkIfTableIsInDatabase(tableName)
            if not tablePresent:
                print('Creating Table...')
                self.databaseModule().createTable(wForCoord, tableName)
            else:
                print('Appending Table to the exiting one...')
                self.databaseModule().appendDataToExistingTable(wForCoord, tableName, drop_duplicates=True, dask=dask)
            # See The database Statistics
            dataForLogs = self.databaseModule().getTableStatisticsFromQuery(tableName, ['date'])
            # Adding logs to see minimum and maximum date
            min_date = self.databaseModule().executeQuery('SELECT MIN(date) FROM public."' + tableName + '"').iloc[0, 0]
            min_date = pd.to_datetime(min_date, format='%Y-%m-%dT%H:%M')
            max_date = self.databaseModule().executeQuery('SELECT MAX(date) FROM public."' + tableName + '"').iloc[0, 0]
            max_date = pd.to_datetime(max_date, format='%Y-%m-%dT%H:%M')

            print('Start Date: ' + str(min_date))
            print('Start Date: ' + str(max_date))

            return wForCoord



