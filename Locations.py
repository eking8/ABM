import networkx as nx
import osmnx as ox
from datetime import timedelta
from functools import lru_cache
import math
import sys
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
        self.fatalities=0 
        self.members=[]
        self.last_conflict_date = None  # Track the date of the last conflict, must be none at start of simulation
        self.waited_days=0
        self.is_open = True
        self.perc=0

    # find lat and long of location
    @staticmethod
    def geocode_location(name, country):
        query = f"{name}, {country}"
        print(f"Geocoding {query}...")
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


    # method to add connections
    def add_connection(self, other_location):
        if not all([self.latitude, self.longitude, other_location.latitude, other_location.longitude]):
            print(f"Missing coordinates for connection between {self.name} and {other_location.name}")
            return
        print(f"Calculating distance between {self.name} and {other_location.name}...")
        distance = self.haversine_distance(self.latitude, self.longitude, other_location.latitude, other_location.longitude)
        # check for border crossings
        crosses_border = self.country != other_location.country
        self.connections.append({'location': other_location, 'distance': distance, 'crosses_border': crosses_border})
    
    def addmember(self,id):
        self.members.append(id)
        self.population += 1

    def removemember(self,id,Agents):
        if id in self.members:
            self.members.remove(id)
            self.population -= 1
        else:
            print(str(id) + " not in " + str(self.name))
            print(self.members)
            print([x.location for x in Agents[id].group])
            print([x.id for x in Agents[id].group if x.is_leader])
            print(Agents[id].instratgroup)
            print([x.id for x in Agents[id].group])
            sys.exit(1)

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
    
    def check_and_update_conflict_status(self, current_date):
        """
        Check and update the city's conflict status based on the last conflict date and the current date.

        Parameters:
        - current_date (datetime): The current date in the simulation.
        """
        if self.hasconflict and self.last_conflict_date:
            if current_date - self.last_conflict_date > timedelta(days=10): #10 day cool down period after event, can be adjusted in SA
                self.hasconflict = False 

    
    def form_strat_group(self,Agents):

        group_size=0
        group=[1]



        for id in self.members:
                
                if not Agents[id].ingroup:
                    if self.perc > 0.09: # Roughly 10% travel in some form of strategic group
                        break
                    elif len(group)>=group_size:
                        for agent in group:
                            if agent!=1:
                                agent.group=group
                        Agents[id].is_leader=True
                        group_size=abs(random.normalvariate(10,4))
                        Agents[id].ingroup=True
                        Agents[id].instratgroup=True
                        group=[Agents[id]]
                    else:
                        Agents[id].ingroup=True
                        Agents[id].instratgroup=True
                        Agents[id].is_leader=False
                        group.append(Agents[id])
                        
                                

        


    
    def perc_in_group(self,Agents):
        ingroup = len([Agents[x] for x in self.members if x.instratgroup])
        total = len([Agents[x] for x in self.members])
        self.perc=ingroup/total
            
        



class City(Location):
    def __init__(self, name, country="Mali", population=None, hasairport=False, top20 = True):
        super().__init__(name, country)
        self.population = population
        self.hasairport = hasairport
        self.iscity = True # by definition
        self.iscamp = False # by definition
        self.top20 = top20 # if has top 20 population


class Camp(Location):
    def __init__(self, name, country, population=None): # Need to change capacity to empirically derived
        super().__init__(name, country)
        self.population = population
        self.hasairport = False # by definition
        self.iscity = False # by definition
        self.iscamp = True # by definition
        

    
def total_pop(cities):
    return sum(city.population for city in cities)

