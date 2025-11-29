# Simple user-getter class to handle both cloud and local environment

def user_getter(user):
    if user == "local":
        return "D:\\PythonProjects-Storage\\WeatherForecast\\"
    elif user == "colab-drive":
        return "/content/drive/MyDrive/WeatherForecast/"
    else:
        raise Exception("User " + str(user) + " not implemented!")

# This is only valid for .csv file
def csv_path_getter(path):
    if path == "full":
        return "weatherForecast"
    elif user == "reduced":
        return "weatherForecast_reduced"
    else:
        raise Exception("User " + str(user) + " not implemented!")