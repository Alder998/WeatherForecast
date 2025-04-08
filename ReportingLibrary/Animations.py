# Animation library to see the evolution of weather data on a selected timestep

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from DatabaseManager import Database as db
from dotenv import load_dotenv
import os

class Animations:

    def __init__(self):
        pass

    # Utils-like function to access the database faster
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

    def generateAnimationOnWeatherVariable (self, grid_step, weatherVariable, colorScale, start_date = None,
                                            end_date = None, save = False):

        # Params
        print('Generating Animation...')
        colorScale = colorScale
        variable = weatherVariable

        # Load Data from Database
        print('Loading data...')
        # Handle it with query
        data = self.databaseModule().executeQuery('SELECT * FROM public."WeatherForRegion_' + str(grid_step) +
                                             '" WHERE date BETWEEN ' + "'" + start_date + "'" + ' AND ' + "'" +
                                             end_date + "'")
        data['date'] = pd.to_datetime(data['date'])
        # Filter for date if necessary
        if (start_date != None) & (end_date != None):
            dt_date_start = pd.to_datetime(start_date, format="%Y-%m-%d").date()
            dt_date_end = pd.to_datetime(end_date, format="%Y-%m-%d").date()
            data = data[(data['date'].dt.date > dt_date_start) &
                        (data['date'].dt.date < dt_date_end)].reset_index(drop = True)

        # Load the region
        print('Loading Plot...')
        world = gpd.read_file(
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson")
        italy = world[world["NAME"] == "Italy"]

        # Configure the figure
        fig, ax = plt.subplots(figsize=(8, 10))
        italy.plot(ax=ax, color='skyblue', edgecolor='black')

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        # Get the range Variable
        vmin, vmax = data[variable].min(), data[variable].max()

        # Normalizer for color bar
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Get scatter for normalization (empty, it will be filled with the update function)
        sc = ax.scatter([], [], c=[], cmap=colorScale, s=50, norm=norm)

        # Color bar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label(variable, fontsize=12)

        # Unique timestamps in data
        time_steps = sorted(data['date'].unique())

        # Function to update the data
        def update(frame):
            current_time = time_steps[frame]
            filtered = data[data['date'] == current_time]

            sc.set_offsets(filtered[['longitude', 'latitude']].values)
            sc.set_array(filtered[variable].values)

            ax.set_title(f"{variable}: {current_time.strftime('%d/%m/%Y %H:%M')}", fontsize=14)

            return sc,

        # Create the animation
        print('Creating Animated Plot on variable: ' + weatherVariable + '...')
        ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=100, repeat=True)
        if save:
            ani.save(r"C:\Users\alder\Desktop\Projects\Wheater Forecast\meteo_" + variable + "_animation.gif",
                     writer="pillow", fps=15)

        # Show
        plt.show()

    def generateAnimationOnWeatherVariableFromDataFrame (self, dataFrame, weatherVariable, colorScale, start_date = None,
                                            end_date = None, save = False):

        # Params
        print('Generating Animation...')
        colorScale = colorScale
        variable = weatherVariable

        # Load Data from Database
        print('Loading data...')
        # Handle it with query
        data = dataFrame
        data['date'] = pd.to_datetime(data['date'])
        # Filter for date if necessary
        if (start_date != None) & (end_date != None):
            dt_date_start = pd.to_datetime(start_date, format="%Y-%m-%d").date()
            dt_date_end = pd.to_datetime(end_date, format="%Y-%m-%d").date()
            data = data[(data['date'].dt.date > dt_date_start) &
                        (data['date'].dt.date < dt_date_end)].reset_index(drop = True)

        # Load the region
        print('Loading Plot...')
        world = gpd.read_file(
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson")
        italy = world[world["NAME"] == "Italy"]

        # Configure the figure
        fig, ax = plt.subplots(figsize=(8, 10))
        italy.plot(ax=ax, color='skyblue', edgecolor='black')

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        # Get the range Variable
        vmin, vmax = data[variable].min(), data[variable].max()

        # Normalizer for color bar
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Get scatter for normalization (empty, it will be filled with the update function)
        sc = ax.scatter([], [], c=[], cmap=colorScale, s=50, norm=norm)

        # Color bar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label(variable, fontsize=12)

        # Unique timestamps in data
        time_steps = sorted(data['date'].unique())

        # Function to update the data
        def update(frame):
            current_time = time_steps[frame]
            filtered = data[data['date'] == current_time]

            sc.set_offsets(filtered[['longitude', 'latitude']].values)
            sc.set_array(filtered[variable].values)

            ax.set_title(f"{variable}: {current_time.strftime('%d/%m/%Y %H:%M')}", fontsize=14)

            return sc,

        # Create the animation
        print('Creating Animated Plot on variable: ' + weatherVariable + '...')
        ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=100, repeat=True)
        if save:
            ani.save(r"C:\Users\alder\Desktop\Projects\Wheater Forecast\meteo_" + variable + "_animation.gif",
                     writer="pillow", fps=15)

        # Show
        plt.show()