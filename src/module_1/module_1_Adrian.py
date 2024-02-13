import requests
import time
from collections import defaultdict
import numpy as np


API_URL = "https://climate-api.open-meteo.com/v1/climate?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"

def call_api(url, params):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as errh:
        print("Http Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
        time.sleep(1)  # simple cool-off
        return call_api(url, params)  # retry
    except requests.exceptions.RequestException as err:
        print("OOps: Something Else", err)

from datetime import datetime, timedelta

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_data_meteo_api(city, start_date, end_date):
    data = []
    for single_date in daterange(start_date, end_date):
        formatted_date = single_date.strftime("%Y-%m-%d")
        params = {
            "latitude": COORDINATES[city]["latitude"],
            "longitude": COORDINATES[city]["longitude"],
            "variables": VARIABLES,
            "start": formatted_date,
            "end": formatted_date
        }
        daily_data = call_api(API_URL, params)
        if daily_data:
            data.append(daily_data)
    return data






def main():
 # Example usage
 start_date = datetime(1950, 1, 1) 
 end_date = datetime(1950, 1, 5)
 madrid_data = get_data_meteo_api("Madrid", start_date, end_date)
 print(madrid_data)
 print("variables")
 


if __name__ == "__main__":
 main()