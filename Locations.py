import numpy as np
import osmnx as ox
from datetime import timedelta
from functools import lru_cache
import math
import random

# store as cache to improve speed on repeated use...

geocode_cache = {}

class Location: 
    def __init__(self, name, country="Mali"):
        self.name = name
        self.country = country
        self.latitude, self.longitude = self.geocode_location(name, country)
        self.connections = []
        self.hasconflict = False
        self.fatalities=0 # initially 0

    # find lat and long of location
        
    @staticmethod
    def geocode_location(name, country):
        query = f"{name}, {country}"
        # print(f"Geocoding {query}...")
        if query in geocode_cache:
            return geocode_cache[query]
        else:
            try:
                lat, lon = ox.geocoder.geocode(query)
                geocode_cache[query] = (lat, lon)
                return lat, lon
            except Exception as e:
                print(f"Geocoding failed for {query} with error {e}")
                return None, None
    
    # find distance between two points on a sphere
            
    @staticmethod
    @lru_cache(maxsize=None)
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c

        return round(distance, 1)
    
    def add_population(self):
        self.population += 1

    def reduce_population(self):
        self.population -= 1

    # method to add connections

    def add_connection(self, other_location):
        if not all([self.latitude, self.longitude, other_location.latitude, other_location.longitude]):
            print(f"Missing coordinates for connection between {self.name} and {other_location.name}")
            return
        # print(f"Calculating distance between {self.name} and {other_location.name}...")
        distance = self.haversine_distance(self.latitude, self.longitude, other_location.latitude, other_location.longitude)
        # check for border crossings
        crosses_border = self.country != other_location.country
        self.connections.append({'location': other_location, 'distance': distance, 'crosses_border': crosses_border})



class City(Location):
    def __init__(self, name, country="Mali", population=None, hasairport=False, top20 = True):
        super().__init__(name, country)
        self.population = population
        self.hasairport = hasairport
        self.iscity = True # by definition
        self.iscamp = False # by definition
        self.last_conflict_date = None  # Track the date of the last conflict, must be none at start of simulationm
        self.top20 = top20 # if has top 20 population
        self.capacity=np.inf # may be changed later

    def in_city_conflict(self, fatalities, current_date):
        """
        Update the conflict status of the city and track the date of the last conflict.

        Parameters:

        - Number of fatalities
        - current_date (datetime): The current date in the simulation.
        """

        self.hasconflict = True # by definition
        self.last_conflict_date = current_date 
        self.fatalities += fatalities # cum fatalitlies

        # print("Conflict updated in " + self.name)

    def check_and_update_conflict_status(self, current_date):
        """
        Check and update the city's conflict status based on the last conflict date and the current date.

        Parameters:
        - current_date (datetime): The current date in the simulation.
        """
        if self.hasconflict and self.last_conflict_date:
            if current_date - self.last_conflict_date > timedelta(days=10): #10 day cool down period after event, can be adjusted in SA
                self.hasconflict = False 
                # print("Conflict removed in " + self.name)
            # else:
                # print("Conflict checked in " + self.name)



class Camp(Location):
    def __init__(self, name, country, population=None, capacity=10000): # Need to change capacity to empirically derived
        super().__init__(name, country)
        self.population = population
        self.hasairport = False # by definition
        self.iscity = False # by definition
        self.iscamp = True # by definition
        self.capacity = capacity

        """
        Select a key from the distances dictionary using roulette method,
        where shorter distances have higher probabilities of being selected.
        
        :param distances: Dictionary of {key: {'distance': value}}
        :return: Selected key based on roulette selection
        """

        if distances is None:
            print('No routes')
            return None
        
        distances = {key: value for key, value in distances.items() if value['distance'] != 0}
            
        
        # Invert distances to treat smaller distances as larger for selection
        inverted_distances = {key: 1.0 / distances[key]['distance'] for key in distances}
        
        # Calculate total sum of inverted distances
        total_inverted_sum = sum(inverted_distances.values())

        if total_inverted_sum == 0:
            return None
        
        # Normalize inverted distances to probabilities
        probabilities = {key: inverted_distances[key] / total_inverted_sum for key in inverted_distances}
        
        # Prepare for roulette wheel selection
        cumulative_prob = 0.0
        cumulative_probs = []
        keys_sorted = sorted(probabilities.keys())  # Ensure consistent order
        for key in keys_sorted:
            cumulative_prob += probabilities[key]
            cumulative_probs.append((cumulative_prob, key))
        
        # Generate a random number and select key based on cumulative probabilities
        r = random.random()
        for cumulative_prob, key in cumulative_probs:
            if r <= cumulative_prob:
                return key
                
        # In case of rounding errors, return the last key
        return keys_sorted[-1]
    

def total_pop(cities):
    sum(city.population for city in cities)

def city_probabilities(cities,total_population):
    city_probabilities = {city.name: city.population / total_population for city in cities}
    prob_values = list(city_probabilities.values())

    # Normalize the probabilities
    normalized_prob_values = [float(i)/sum(prob_values) for i in prob_values]

    if not np.isclose(sum(normalized_prob_values), 1.0):
        raise ValueError("Normalized probabilities do not sum closely to 1.")
    
    return (city_probabilities,normalized_prob_values)