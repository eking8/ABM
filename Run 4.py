"""

TO DO

- Logic for passing No-Go list to 1-3 random agents per node
- Ensure for the above that the random agents are selected from not in family
- Logic for creating families (household size distribtuons)
- Families will share the following:
----- 'No-Go' zones  
----- Slowest walking speed
- Followers (such as children) should not follow same decision logic (if statement at start which sees if follower)
- Consider group size distributions and begin to model stategic groups
- Logic for reassessing at each node the group
- Consider distance form conflicts as an attribute for cities (will help to model IDPs)
- "DANGER LEVELS' -> roulette function now a function of a score rather than just distance
- Conflict fatalieis -> agent death mechanisms
- Agent birth mechanisms
- Rich leaving the airport to an 'abroad' index
- Incorporate average times at different nodes
- Foreign conflicts

SCORE BASED ON:
- Destination node distance
- How frequent the link is travelled
- Destination node danger
- First node in path danger
- What people are saying (i.e. other people's destination)
- Perhaps percentage full


COMMUNICATION
- Don't focus as much on the knowledge of routes and settlements
- Focus on the perception of danger
- i.e. an agent communicating with another agent about a dange level of node/link
"""




import geopandas as gpd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import math
from datetime import datetime, timedelta
from functools import lru_cache
import numpy as np
import random
import csv
import time

# Colours for printed text

class colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    END = '\033[0m'

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

    
def create_graph(locations):

    """
    
    This function allows an initial formation of the network with all non-time-dependent features to be added
    
    Parameters:
    - locations (list)

    """
    # print("Initialising graph...")

    G = nx.Graph()

    for loc in locations:
        # Determine node type based on properties
        node_type = 'City' if hasattr(loc, 'iscity') and loc.iscity else 'Camp'
        # Add node with initial attributes except for conflict
        G.add_node(loc.name, pos=(loc.longitude, loc.latitude), type=node_type, has_airport=hasattr(loc, 'hasairport') and loc.hasairport, has_conflict=loc.hasconflict, country=loc.country, fatalities=loc.fatalities, 
                   population=loc.population, capacity=loc.capacity)
        
        # Add edges with border crossing information
        for conn in loc.connections:
            G.add_edge(loc.name, conn['location'].name, weight=round(conn['distance'],3), crosses_border=conn['crosses_border'])

    # print("Graph initialised.")
    
    return G
    
    

def draw_graph(G, current_date, distances_on=False):
    
    """
    
    This function builds on graph and draws a current state of the conflict zones
    
    iparameters:
    - G (networkx.Graph)
    - current_date (datetime)
    
    """

    # print(f"Drawing network as of {current_date}...")
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(15, 10))
    
    # Variables to help differentiate node types and conflicts
    colors = {'default': 'gray', 'City': 'blue', 'Camp': 'green', 'Airport': 'cyan'}
    shapes = {'City': 's', 'Camp': '^', 'Airport': 'o'}

    
    
    # Adjust node colors based on real-time conflict data
    for node, attr in G.nodes(data=True):

        if attr['has_conflict']==True:
            color = 'red'
        elif attr['type'] == 'City' and attr.get('has_airport', False):
            color = colors['Airport'] 
        else:
            color = colors.get(attr['type'], 'default')

        shape = shapes.get(attr['type'], 'o')

        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, node_shape=shape, node_size=100)

    
    
    # Draw edges, differentiating border crossings
    for u, v, d in G.edges(data=True):
        style = 'dotted' if d.get('crosses_border', False) else 'solid'
        
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='gray', style=style)

    if distances_on:
        weights = nx.get_edge_attributes(G, 'weight')
        edge_labels = {(u, v): d for (u, v), d in weights.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    nx.draw_networkx_labels(G, pos, font_size=8, verticalalignment='bottom')
    plt.title(f"Network of Cities, Camps, and Airports as of {current_date}")
    plt.axis('on')
    plt.show()
    # print("Network visualization completed.")

def find_accessible_nodes_within_distance(G, start_node, max_distance_km):
    """
    Finds all nodes within a specified distance of a start node in a graph.
    
    Parameters:
    - G (networkx.Graph): The graph representing the network.
    - start_node (str): The name of the starting node.
    - max_distance_km (float): The maximum distance (in kilometers) to search within.
    
    Returns:
    - List[str]: A list of node names within the specified distance.
    """
    visited = set()  # To keep track of visited nodes to avoid revisiting
    accessible_nodes = []  # To store nodes within the specified distance
    
    # Queue for BFS, storing tuples of (node, distance_from_start)
    queue = [(start_node, 0)]
    
    while queue:
        current_node, current_distance = queue.pop(0)
        
        # If this node is within the max distance and not visited, add it
        if current_node not in visited and current_distance <= max_distance_km:
            visited.add(current_node)
            accessible_nodes.append(current_node)
            
            # Add neighbors to the queue
            for neighbor, attributes in G[current_node].items():
                edge_distance = attributes['weight']
                if current_distance + edge_distance <= max_distance_km:
                    queue.append((neighbor, current_distance + edge_distance))
        
    
    accessible_nodes.remove(start_node)

    if len(accessible_nodes)==0:
        return None
    
    return accessible_nodes


class Agent:
    # Age groups and their population
    age_groups = [(90, 100), (80, 89), (70, 79), (60, 69), (50, 59), (40, 49), (30, 39), (20, 29), (10, 19), (0, 9)]
    populations = [21510, 88052, 289198, 632542, 984684, 1569836, 2451989, 3466640, 5240091, 7650947]

    probabilities = []
    cumulative_distribution = []

    @classmethod
    def initialise_cities(cls,cities):
        cls.foreign_cities = []
        for city in cities:
            cls.foreign_cities.append(city.name)


    @classmethod
    def calculate_distributions(cls):
        cls.probabilities = [pop / sum(cls.populations) for pop in cls.populations]
        cls.cumulative_distribution = np.cumsum(cls.probabilities)

    def __init__(self, id, status = 'Resident'):
        self.id = id
        self.age = self.generate_random_age()
        self.gender = self.generate_gender()
        self.status = status
        self.location = np.random.choice(
            list(city_probabilities.keys()),
            p=normalized_prob_values
        )
        self.shortterm = self.location  # iniitally no plan
        self.longterm = None  # Intended destination
        self.moving = False  # Is the agent currently moving to a destination
        self.threshold = np.random.uniform(0,30)  # InitiaClise the threshold here or in a separate method
        rand_n= np.random.uniform(0,1)
        self.startdate=None
        self.enddate=None
        self.traveltime=None
        self.nogos=[]
        self.city_origin = self.location
        self.country_origin='Mali'
        self.is_stuck=False
        self.been_abroad=False
        
        self.is_leader=False 
        self.in_family=False # Replace with statistical distribution
        self.fam_size=1
        self.fam=None
        self.is_leader=True

        if self.in_family:
            self.fam_size=3 # Replace with statistical distribution
            self.fam=[] # Assign other agents 
            self.speed=min([x.speed for x in self.fam])
            # self.is_leader === >logic that randomly assigns leader
            



        if rand_n>0.90:
            self.capitalbracket = 'Rich'
        elif rand_n>0.50:
            self.capitalbracket = 'Mid'
        else:
            self.capitalbracket = 'Poor'

        if self.capitalbracket == 'Poor': # No access to car
            self.speed = abs(np.random.normal(200,10))
        else:
            self.speed = abs(np.random.normal(600, 10))
        

    # Calculate the cumulative distribution
    cumulative_distribution = np.cumsum(probabilities)

    def generate_random_age(self):
        # Generate a random float in the range [0, 1)
        rand = np.random.random()
        
        # Find the age group corresponding to the random number
        for i, threshold in enumerate(self.__class__.cumulative_distribution):
            if rand < threshold:
                # Return a random age within the selected age group
                return np.random.randint(self.__class__.age_groups[i][0], self.__class__.age_groups[i][1] + 1)

    @staticmethod
    def generate_gender():
        probabilities = [0.503, 0.497]
        choices = ['M', 'F']
        return random.choices(choices, weights=probabilities, k=1)[0]

    def assess_situation_and_move_if_needed(self,city,current_date):
        self.location = self.shortterm
        if city.hasconflict and city.fatalities > self.threshold:
            if self.status == 'Resident':
                    self.moving = True
                    self.status = 'Fleeing from conflict'
                    # print(colors.RED + "Agent " + str(self.id) + " is now fleeing from " + str(self.location) + colors.END)
                    self.startdate=current_date


        

        if self.moving == True and self.status in ['Refugee','Returnee','IDP','Fleeing from conflict']:

            if self.location in self.__class__.foreign_cities:
                self.status='Refugee'
                self.been_abroad=True
            elif self.been_abroad:
                self.status='Returnee'
            else:
                self.status='IDP'

            self.traveltime=current_date-self.startdate
            
            if self.is_leader: # followers do not decide where to go

                if self.location == self.longterm:
                    self.moving = False
                    self.status = 'Resident'
                    # print(colors.GREEN + "Agent " + str(self.id) + " has reached " + str(self.longterm) + colors.END)
                    self.enddate=current_date
                    # LOGIC THAT ASSIGNS STATUS AND MOVING CHANGE FOR FAMILY (FOLLOWERS)
                
                else:

                    if self.capitalbracket == 'Rich':
                        des_dic = find_nearest_cities_with_airport(G,self.location,self.speed,self.nogos)

                    elif self.capitalbracket == 'Mid':
                        des_dic=find_shortest_paths_to_neighboring_countries(G,self.location,self.speed,current_date,self.nogos)

                    else:
                        des_dic=camp_paths(G,self.location,self.speed,current_date,self.nogos)
                    
                    if des_dic:

                            key = self.roulette_select(des_dic)

                            if key:
                                    
                                self.distanceleft=des_dic[key]['distance']
                                self.longterm=key
                                self.shortterm=des_dic[key]['path'][0]
                                # print("location = " + self.location + ", and short term = " + str(self.shortterm))
                                # print(colors.YELLOW + "Agent " + str(self.id) + " is going to the camp in " + str(self.longterm) + " from "+ str(self.location) + colors.END)
                                # LOGIC THAT ASSIGNS STATUS AND MOVING CHANGE FOR FAMILY (FOLLOWERS)

                            else:

                                # print("Distance too large for Agent " + str(self.id) + " to travel")
                                self.is_stuck=True
                                self.moving=False
                                # LOGIC THAT ASSIGNS STUCK AND MOVING CHANGE FOR FAMILY (FOLLOWERS)
            
        
        


    
    def roulette_select(self, distances):
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
            
            

            

                      
                    

                
    

    



def camp_paths(G, start_node,max_link_length,current_date,nogo):
    """
    Finds shortest paths and distances from a given start node to all camps in the graph.

    Parameters:
    - G (networkx.Graph): The graph of locations.
    - start_node (str): The name of the starting node.

    Returns:
    - dict: A dictionary mapping camp names to a tuple containing the shortest path and its distance.

    Future considerations of parameters to be included in G_filtered:
    - No-zones: conflicts or areas cited as dangerous through rumours that should be avoided
    - Go-zones: areas cited as safe by rumours

    """
    # Filter edges based on max_link_length
    G_filtered = filter_graph_by_max_link_length(G, max_link_length,start_node,nogo=nogo)

    if not G_filtered.has_node(start_node):
        # The start node is not present in G_filtered. It might be separated from the rest of the graph
        # This could mean that the walk speed is too slow to leave
        print(max_link_length,G_filtered,start_node)
        print('error with start node')
        return None
    
    distances, paths = nx.single_source_dijkstra(G_filtered, start_node)
    camp_paths_distances = {}
    for node, data in G.nodes(data=True):
        if G.nodes[node].get('type') == 'Camp' and node in paths:
            if data.get('country',None) in ['Niger','Burkina Faso'] and current_date < datetime(2012, 2, 21).date():
                pass # boarder does not open before 21st feb
            else:
                paths[node].remove(start_node)
                camp_paths_distances[node] = {'path':paths[node], 'distance':distances[node]}

    
    return camp_paths_distances


def find_nearest_cities_with_airport(G, start_node, max_link_length,nogo):
    """
    Finds the 5 nearest cities with an airport from the given start node, considering a maximum link length,
    and outputs a dictionary where each key is a city name and each value is a dictionary containing the distance and path.
    
    Parameters:
    - G (networkx.Graph): The graph representing the network.
    - start_node (str): The name of the starting node.
    - max_link_length (float): Maximum acceptable link length in kilometers.
    
    Returns:
    - A dictionary with the 5 nearest cities' names as keys, each associated with a dictionary containing 'distance' and 'path'.
    """
    G_filtered = filter_graph_by_max_link_length(G, max_link_length,start_node,nogo=nogo)

    if not G_filtered.has_node(start_node):
        print('Error: Start node is not present in the filtered graph.')
        return {}

    cities_info = {}

    for node in G_filtered.nodes:
        if node != start_node and G_filtered.nodes[node].get('has_airport', False):
            try:
                path_length = nx.shortest_path_length(G_filtered, start_node, node, weight='weight')
                path = nx.shortest_path(G_filtered, start_node, node, weight='weight')[1:]  # Exclude start node from path
                cities_info[node] = {'distance': path_length, 'path': path}
            except nx.NetworkXNoPath:
                pass

    # Sort the cities by distance and select the top 5
    nearest_cities = dict(sorted(cities_info.items(), key=lambda item: item[1]['distance'])[:5])

    return nearest_cities



def find_shortest_paths_to_neighboring_countries(G, start_node, max_link_length, current_date,nogo):
    """
    Finds the shortest distance and respective paths from the start node to the closest city 
    in each neighboring country, considering a maximum link length and specific border closures.
    The function specifically checks for border closures with Niger and Burkina Faso before February 21, 2012.
    
    Parameters:
    - G (networkx.Graph): The graph representing the network of locations.
    - start_node (str): The name of the starting node/city.
    - max_link_length (float): Maximum acceptable link length in kilometers.
    - current_date (datetime.date): The current date for considering dynamic conditions such as border openings.
    
    Returns:
    - dict: A dictionary where keys are city names (of closest cities in neighboring countries), 
      and values are dictionaries with 'distance' and 'path'.
    """
    G_filtered = filter_graph_by_max_link_length(G, max_link_length,start_node,nogo=nogo)

    if not G_filtered.has_node(start_node):
        print('Error: Start node is not present in the filtered graph.')
        return {}

    results = {}
    start_country = G.nodes[start_node].get('country', 'Not specified')

    for node, data in G_filtered.nodes(data=True):
        country = data.get('country', None)
        if country and country != start_country:
            # Apply the border closure rule for Niger and Burkina Faso
            if country in ['Niger', 'Burkina Faso'] and current_date < datetime(2012, 2, 21).date():
                continue  # Skip this city if the current date is before the border opening date

            try:
                path_length = nx.shortest_path_length(G_filtered, start_node, node, weight='weight')
                path = nx.shortest_path(G_filtered, start_node, node, weight='weight')

                # If the city is closer than any previously found or if it's the first city found for this country
                if node not in results or path_length < results[node]['distance']:
                    results[node] = {'distance': path_length, 'path': path}
            except nx.NetworkXNoPath:
                pass  # No path exists to this node, ignore it

    return results


def filter_graph_by_max_link_length(G, max_link_length, start_node, nogo= []):
    # Step 1: Create a new empty graph to hold the filtered graph
    G_filtered = nx.Graph()

    if start_node in nogo:
        nogo.remove(start_node)
    
    # Step 2: Copy all nodes from G to G_filtered, preserving attributes
    for node, data in G.nodes(data=True):
        
        if node not in nogo:
            if data.get('population', False) < data.get('capacity', False):
                G_filtered.add_node(node, **data)
    
    # Step 3: Filter edges by max_link_length and add them to G_filtered
    for u, v, d in G.edges(data=True):
        if d.get('weight', float('inf')) <= max_link_length:
            G_filtered.add_edge(u, v, **d)
    
    return G_filtered

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



# populations from 2009 census https://www.citypopulation.de/en/mali/cities/

bamako = City("Bamako", hasairport=True, population = 1810366)
sikasso = City("Sikasso", hasairport=True, population = 226618)
koutiala = City("Koutiala", hasairport=True, population = 141444)
segou = City("Segou", population = 0.0321*133501)
kayes = City("Kayes", hasairport=True, population = 126319)
gao = City("Gao", hasairport=True, population = 86353)
san = City("San", population = 66967)
bougouni = City("Bougouni", population = 58538)
tombouctou = City("Timbuktu", hasairport=True, population = 54629)
kita = City("Kita", population = 49043)

niono_socoura= City("Niono", population = 34113)
mopti = City("Mopti", hasairport=True, population=120786)
koulikoro = City("Koulikoro", population=41602)
fana = City("Fana", population=36854)
nioro = City("Nioro", hasairport=True, population=33691)
kidal = City("Kidal", population=25969)
douentza = City("Douentza", hasairport=True, population=24005)
kadiolo = City("Kadiolo", population=24749)
djenne = City("Djenne", population=26267)
zegoua = City("Zegoua", population=20496)






cities = [bamako, sikasso, koutiala, segou, kayes, gao, san, bougouni, tombouctou, kita,
          niono_socoura, mopti, koulikoro, fana, nioro, kidal, douentza, kadiolo, djenne, zegoua]


# 2013 census https://www.citypopulation.de/en/senegal/cities/

# Senegal
bakel = City("Bakel", country="Senegal", population=13329)
tambacounda = City("Tambacounda", country="Senegal", population=107293)
kedougou = City("Kedougou", country="Senegal", population=30051)

# 2014 Census https://www.citypopulation.de/en/guinea/cities/

# Guinea
dinguiraye = City("Dinguiraye", country="Guinea", population=18082)
siguiri = City("Siguiri", country="Guinea", population=127492)
fodekaria = City("Fodekaria", country="Guinea", population=20112)
kankan = City("Kankan", country="Guinea", population=190722)
mandiana = City("Mandiana", country="Guinea", population=16460)

# 2014 census https://www.citypopulation.de/en/ivorycoast/cities/

# Côte D'ivoire
odienne = City("Odienne", country="Côte D'ivoire", population=42173)
tingrela = City("Tingrela", country="Côte D'ivoire", population=40323)
boundiali = City("Boundiali", country="Côte D'ivoire", population=39962)
korhogo = City("Korhogo", country="Côte D'ivoire", population=243048)

# Average of 2006 and 2019

# Burkina Faso
banfora = City("Banfora", country="Burkina Faso", population=(75917+117452)/2)
dande = City("Dandé", country="Burkina Faso", population=(0)/2)
solenzo = City("Solenzo", country="Burkina Faso", population=(16850+24783)/2)
nouna = City("Nouna", country="Burkina Faso", population=(22166+32428)/2)
dedougou = City("Dedougou", country="Burkina Faso", population=(38862+63617)/2)
tougan = City("Tougan", country="Burkina Faso", population=(17050+26347)/2)
ouahigouya = City("Ouahigouya", country="Burkina Faso", population=(73153+124587)/2)
arbinda = City("Arbinda", country="Burkina Faso", population=(9790	+31215)/2)
dori = City("Dori", country="Burkina Faso", population=(21078+46512)/2)

# 2012 census https://www.citypopulation.de/en/niger/cities/
# Niger
ayorou = City("Ayorou", country="Niger", population=11528)
tera = City("Tera", country="Niger", population=29119)
filingue = City("Filingué", country="Niger", population=12224)
assamakka = City("Assamakka", country="Niger", population=0)

# 2008 census https://www.citypopulation.de/en/algeria/cities/

# Algeria
bordj_badji_mokhtar = City("Bordj Badji Mokhtar", country="Algeria", population=628475)

# 2013 census https://www.citypopulation.de/en/mauritania/cities/

# Mauritania
walatah = City("Walatah", country="Mauritania", population=0)
nema = City("Néma", country="Mauritania", population=21708)
timbedra = City("Timbédra", country="Mauritania", population=14131)
tchera_rouissa = City("Tchera Rouissa", country="Mauritania", population=0)
adel_bagrou = City("Adel Bagrou", country="Mauritania", population=8576)
tintane = City("Tintane", country="Mauritania", population=12690)
selibaby = City("Sélibaby", country="Mauritania", population=26420)




foreign_cities = [
    bakel, tambacounda, kedougou,
    dinguiraye, siguiri, fodekaria, kankan, mandiana,
    odienne, tingrela, boundiali, korhogo,
    banfora, dande, solenzo, nouna, dedougou, tougan, ouahigouya, arbinda, dori,
    ayorou, tera, filingue, assamakka,
    bordj_badji_mokhtar,
    walatah, nema, timbedra, tchera_rouissa, adel_bagrou, tintane, selibaby
]




# Define camps
bobo = Camp("Bobo", "Burkina Faso", population=0) # 2012-02-29
goudoubo = Camp("Goudoubo", "Burkina Faso",population=0) # 2012-02-29
mentao = Camp("Mentao", "Burkina Faso", population=0) # 2012-02-29
ouagadougou = Camp("Ouagadougou", "Burkina Faso", population=0) # 2012-02-29
fassala = Camp("Fassala", "Mauritania", population= 17790) # 2012-02-29
mbera = Camp("Mbera", "Mauritania", population=0) # 2012-02-29
abala = Camp("Abala", "Niger", population=1175) # 2012-02-29
intikane = Camp("Intikane", "Niger", population=5058) # 2013-05-07
mangaize = Camp("Mangaize", "Niger", population=1110) # 2012-02-29
niamey = Camp("Niamey", "Niger", population=0) # 2012-02-29
tabareybarey = Camp("Tabareybarey", "Niger", population=0) # 2012-02-29

camps = [bobo, goudoubo, mentao, ouagadougou, fassala, mbera, abala, intikane, mangaize, niamey, tabareybarey]

# populations from 2009 census https://www.citypopulation.de/en/mali/cities/

# Define airports
kenieba = City("Kéniéba", hasairport=True, top20=False, population=0)
yelimane = City("Yélimané", hasairport=True, top20=False, population=0)
ansongo = City("Ansongo", hasairport=True, top20=False, population=0)
bafoulabe = City("Bafoulabé", hasairport=True, top20=False, population=0)
goundam = City("Goundam", hasairport=True, top20=False, population=12586)
tessalit = City("Tessalit", hasairport=True, top20=False, population=0)
bourem = City("Bourem", hasairport=True, top20=False, population=27488)
bandiagara = City("Bandiagara", hasairport=True, top20=False, population=17166)
bengassi = City("Bengassi", hasairport=True, top20=False, population=0)
menaka = City("Menaka", hasairport=True, top20=False, population=9138)

airports = [mopti, kenieba, yelimane, menaka, douentza, ansongo, bafoulabe, goundam, tessalit, bourem, bandiagara, bengassi, nioro]




# populations from 2009 census https://www.citypopulation.de/en/mali/cities/ 

# Non-top 20 cities

lere = City("Lere", top20=False, population=0)
syama = City("Syama", top20=False, population=0)
diabaly = City("Diabaly", top20=False, population=0)
sevare = City("Sevare", top20=False, population=0)
nara = City("Nara", top20=False, population=15310)
niafunke = City("Niafunke", top20=False, population=0)
aguelhok = City("Aguelhok", top20=False, population=0)
koue = City("Koue", top20=False, population=0)
sari = City("Sari", top20=False, population=0)
diago = City("Diago", top20=False, population=0)
ber = City("Ber", top20=False, population=0)
anefis = City("Anefis", top20=False, population=0)
dire = City("Dire", top20=False, population=20337)
tenenkou = City("Tenenkou", top20=False, population=0)
youwarou = City("Youwarou", top20=False, population=0)
hombori = City("Hombori", top20=False, population=0)
tin_zaouaten = City("Tinzaouaten", top20=False, population=0)
anderamboukane = City("Anderamboukane", top20=False, population=0)


cities.append(lere)
cities.append(syama)
cities.append(diabaly)
cities.append(sevare)
cities.append(nara)
cities.append(niafunke)
cities.append(aguelhok)
cities.append(koue)
cities.append(sari)
cities.append(diago)
cities.append(ber)
cities.append(anefis)
cities.append(dire)
cities.append(tenenkou)
cities.append(youwarou)
cities.append(hombori)
cities.append(tin_zaouaten)
cities.append(anderamboukane)

cities += airports
# Combine all locations into a single list
locations = cities + camps + foreign_cities


## Following section has been completed manually from inspection

kayes.add_connection(kita)
kayes.add_connection(segou)
kayes.add_connection(bamako)
kita.add_connection(segou)
kita.add_connection(bamako)
kita.add_connection(san)
bamako.add_connection(segou)
bamako.add_connection(san)
bamako.add_connection(koutiala)
bamako.add_connection(bougouni)
bamako.add_connection(sikasso)
bougouni.add_connection(sikasso)
sikasso.add_connection(koutiala)
segou.add_connection(san)
segou.add_connection(koutiala)
segou.add_connection(tombouctou)
segou.add_connection(gao)
koutiala.add_connection(san)
koutiala.add_connection(gao)
san.add_connection(gao)
gao.add_connection(tombouctou)
korhogo.add_connection(zegoua)
korhogo.add_connection(banfora)
korhogo.add_connection(tingrela)
boundiali.add_connection(tingrela)
nioro.add_connection(kayes)
nioro.add_connection(mbera)
nioro.add_connection(fassala)
nioro.add_connection(kita)
nioro.add_connection(koulikoro)
nioro.add_connection(segou)
nioro.add_connection(niono_socoura)
koulikoro.add_connection(kayes)
koulikoro.add_connection(kita)
koulikoro.add_connection(bamako)
koulikoro.add_connection(koutiala)
koulikoro.add_connection(san)
koulikoro.add_connection(segou)
koulikoro.add_connection(niono_socoura)
niono_socoura.add_connection(kayes)
niono_socoura.add_connection(mbera)
niono_socoura.add_connection(fassala)
niono_socoura.add_connection(tombouctou)
niono_socoura.add_connection(fana)
niono_socoura.add_connection(douentza)
niono_socoura.add_connection(mopti)
niono_socoura.add_connection(djenne)
mopti.add_connection(djenne)
mopti.add_connection(segou)
mopti.add_connection(fassala)
mopti.add_connection(fana)
mopti.add_connection(douentza)
mopti.add_connection(goudoubo)
mopti.add_connection(mentao)
mopti.add_connection(ouagadougou)
mopti.add_connection(bobo)
fana.add_connection(fassala)
fana.add_connection(mbera)
fana.add_connection(tombouctou)
fana.add_connection(gao)
fana.add_connection(douentza)
fana.add_connection(mopti)
fana.add_connection(djenne)
fana.add_connection(bamako)
fana.add_connection(koulikoro)
fana.add_connection(sikasso)
fana.add_connection(koutiala)
fana.add_connection(san)
kidal.add_connection(tombouctou)
kidal.add_connection(gao)
kidal.add_connection(intikane)
douentza.add_connection(fassala)
douentza.add_connection(mbera)
douentza.add_connection(fana)
douentza.add_connection(gao)
douentza.add_connection(tabareybarey)
douentza.add_connection(goudoubo)
douentza.add_connection(mentao)
kadiolo.add_connection(zegoua)
kadiolo.add_connection(bougouni)
kadiolo.add_connection(sikasso)
kadiolo.add_connection(koutiala)
kadiolo.add_connection(bobo)
zegoua.add_connection(bobo)
zegoua.add_connection(sikasso)
zegoua.add_connection(koutiala)
sari.add_connection(mopti)
sari.add_connection(hombori)
sari.add_connection(ouahigouya)
sari.add_connection(mentao)
sari.add_connection(arbinda)
sari.add_connection(douentza)
mbera.add_connection(youwarou)
mbera.add_connection(dire)
fassala.add_connection(timbedra)
fassala.add_connection(sevare)
abala.add_connection(ansongo)
abala.add_connection(ayorou)
abala.add_connection(tera)
mangaize.add_connection(ansongo)
mangaize.add_connection(menaka)
mangaize.add_connection(dori)
niamey.add_connection(anderamboukane)
ouagadougou.add_connection(arbinda)
ouagadougou.add_connection(dori)
ouagadougou.add_connection(lere)
mentao.add_connection(lere)
mentao.add_connection(koue)
mentao.add_connection(dedougou)
bobo.add_connection(tingrela)
bobo.add_connection(syama)
bobo.add_connection(fana)
syama.add_connection(sikasso)
tingrela.add_connection(sikasso)
hombori.add_connection(dire)
hombori.add_connection(douentza)
hombori.add_connection(gao)
hombori.add_connection(ansongo)
hombori.add_connection(tabareybarey)
hombori.add_connection(ayorou)
hombori.add_connection(goudoubo)
hombori.add_connection(arbinda)
hombori.add_connection(mentao)
ansongo.add_connection(gao)
anderamboukane.add_connection(ansongo)
anderamboukane.add_connection(menaka)
anderamboukane.add_connection(intikane)
anderamboukane.add_connection(abala)
anderamboukane.add_connection(mangaize)
ouahigouya.add_connection(mentao)
ouahigouya.add_connection(ouagadougou)
ouahigouya.add_connection(tougan)
ouahigouya.add_connection(lere)
lere.add_connection(tougan)
lere.add_connection(san)
sari.add_connection(bandiagara)
arbinda.add_connection(bandiagara)
diabaly.add_connection(mbera)
nioro.add_connection(timbedra)
nema.add_connection(goundam)
nema.add_connection(niafunke)
tin_zaouaten.add_connection(menaka)
tin_zaouaten.add_connection(aguelhok)
tin_zaouaten.add_connection(anefis)
menaka.add_connection(assamakka)
ansongo.add_connection(menaka)
bourem.add_connection(menaka)
hombori.add_connection(niafunke)
ber.add_connection(tombouctou)
ber.add_connection(bourem)
ber.add_connection(gao)
ber.add_connection(anefis)
ber.add_connection(kidal)
ber.add_connection(aguelhok)
bourem.add_connection(menaka)
tera.add_connection(ouagadougou)
dori.add_connection(mentao)
anderamboukane.add_connection(tabareybarey)
arbinda.add_connection(tabareybarey)
youwarou.add_connection(dire)
youwarou.add_connection(douentza)
sevare.add_connection(douentza)
douentza.add_connection(dire)
douentza.add_connection(youwarou)
douentza.add_connection(sevare)
dire.add_connection(goundam)
dire.add_connection(tombouctou)
dire.add_connection(ber)
sevare.add_connection(djenne)
sevare.add_connection(koue)
koue.add_connection(nouna)
djenne.add_connection(san)
lere.add_connection(nouna)
tougan.add_connection(dedougou)
dedougou.add_connection(bobo)
dande.add_connection(banfora)
diago.add_connection(kita)
diago.add_connection(koulikoro)
diago.add_connection(san)
sari.add_connection(tabareybarey)
sari.add_connection(ayorou)
ouahigouya.add_connection(niamey)
tera.add_connection(filingue)
tera.add_connection(mangaize)
youwarou.add_connection(hombori)
douentza.add_connection(ayorou)
hombori.add_connection(anderamboukane)
kayes.add_connection(nara)
nara.add_connection(tchera_rouissa)
nara.add_connection(adel_bagrou)
nara.add_connection(mbera)
nara.add_connection(fassala)
dire.add_connection(goundam)
dire.add_connection(niafunke)
niafunke.add_connection(goundam)
niafunke.add_connection(youwarou)
niafunke.add_connection(mbera)
niafunke.add_connection(fassala)
youwarou.add_connection(fassala)
youwarou.add_connection(diabaly)
youwarou.add_connection(tenenkou)
youwarou.add_connection(sevare)
youwarou.add_connection(mopti)
diabaly.add_connection(nara)
diabaly.add_connection(niono_socoura)
diabaly.add_connection(fassala)
diabaly.add_connection(tenenkou)
tenenkou.add_connection(djenne)
tenenkou.add_connection(niono_socoura)
tenenkou.add_connection(fassala)
tenenkou.add_connection(sevare)
tenenkou.add_connection(bandiagara)
sevare.add_connection(mopti)
sevare.add_connection(bandiagara)
koue.add_connection(san)
koue.add_connection(djenne)
koue.add_connection(lere)
koue.add_connection(bandiagara)
lere.add_connection(san)
lere.add_connection(bandiagara)
diago.add_connection(fana)
diago.add_connection(bamako)
diago.add_connection(koutiala)
bengassi.add_connection(kenieba)
bengassi.add_connection(tambacounda)
bengassi.add_connection(bafoulabe)
bengassi.add_connection(kayes)
kenieba.add_connection(kedougou)
bafoulabe.add_connection(tambacounda)
mbera.add_connection(tombouctou)
mbera.add_connection(fassala)
fassala.add_connection(kayes)
fassala.add_connection(tombouctou)
bobo.add_connection(bougouni)
bobo.add_connection(sikasso)
bobo.add_connection(koutiala)
bobo.add_connection(ouagadougou)
ouagadougou.add_connection(sikasso)
ouagadougou.add_connection(koutiala)
ouagadougou.add_connection(san)
ouagadougou.add_connection(mentao)
ouagadougou.add_connection(goudoubo)
ouagadougou.add_connection(niamey)
niamey.add_connection(san)
niamey.add_connection(mentao)
niamey.add_connection(goudoubo)
niamey.add_connection(tabareybarey)
niamey.add_connection(mangaize)
niamey.add_connection(abala)
mentao.add_connection(segou)
mentao.add_connection(san)
mentao.add_connection(goudoubo)
goudoubo.add_connection(tabareybarey)
goudoubo.add_connection(mangaize)
mangaize.add_connection(tabareybarey)
mangaize.add_connection(abala)
tabareybarey.add_connection(abala)
tabareybarey.add_connection(intikane)
tabareybarey.add_connection(gao)
abala.add_connection(gao)
abala.add_connection(intikane)
intikane.add_connection(gao)
yelimane.add_connection(nioro)
yelimane.add_connection(kayes)
yelimane.add_connection(koulikoro)
bafoulabe.add_connection(kayes)
bafoulabe.add_connection(kita)
bengassi.add_connection(kita)
kenieba.add_connection(kita)
kenieba.add_connection(bamako)
goundam.add_connection(mbera)
goundam.add_connection(fassala)
goundam.add_connection(tombouctou)
goundam.add_connection(douentza)
bourem.add_connection(tombouctou)
bourem.add_connection(gao)
bourem.add_connection(kidal)
menaka.add_connection(kidal)
menaka.add_connection(gao)
menaka.add_connection(intikane)
ansongo.add_connection(tabareybarey)
menaka.add_connection(gao)
menaka.add_connection(douentza)
bandiagara.add_connection(mopti)
bandiagara.add_connection(segou)
bandiagara.add_connection(djenne)
bandiagara.add_connection(mentao)
bandiagara.add_connection(niono_socoura)
kenieba.add_connection(bengassi)
kenieba.add_connection(kedougou)
kenieba.add_connection(tambacounda)
kenieba.add_connection(dinguiraye)
bengassi.add_connection(tambacounda)
bengassi.add_connection(bafoulabe)
selibaby.add_connection(yelimane)
selibaby.add_connection(nioro)
selibaby.add_connection(tintane)
selibaby.add_connection(bakel)
bakel.add_connection(kayes)
bakel.add_connection(bafoulabe)
bakel.add_connection(tambacounda)
tambacounda.add_connection(yelimane)
tambacounda.add_connection(kayes)
kedougou.add_connection(kayes)
kedougou.add_connection(kita)
kedougou.add_connection(bamako)
kedougou.add_connection(dinguiraye)
kedougou.add_connection(tambacounda)
dinguiraye.add_connection(bamako)
dinguiraye.add_connection(kita)
dinguiraye.add_connection(siguiri)
dinguiraye.add_connection(fodekaria)
dinguiraye.add_connection(kankan)
siguiri.add_connection(bamako)
siguiri.add_connection(bougouni)
siguiri.add_connection(fodekaria)
fodekaria.add_connection(bougouni)
fodekaria.add_connection(kankan)
fodekaria.add_connection(mandiana)
kankan.add_connection(mandiana)
mandiana.add_connection(odienne)
mandiana.add_connection(tingrela)
mandiana.add_connection(boundiali)
mandiana.add_connection(bougouni)
odienne.add_connection(tingrela)
odienne.add_connection(boundiali)
tingrela.add_connection(bougouni)
tingrela.add_connection(syama)
tingrela.add_connection(kadiolo)
tingrela.add_connection(zegoua)
syama.add_connection(kadiolo)
syama.add_connection(bougouni)
syama.add_connection(banfora)
odienne.add_connection(sikasso)
odienne.add_connection(kadiolo)
odienne.add_connection(zegoua)
odienne.add_connection(boundiali)
odienne.add_connection(korhogo)
boundiali.add_connection(korhogo)
boundiali.add_connection(zegoua)
boundiali.add_connection(banfora)
banfora.add_connection(zegoua)
banfora.add_connection(kadiolo)
banfora.add_connection(bobo)
dande.add_connection(bobo)
dande.add_connection(sikasso)
dande.add_connection(koutiala)
dande.add_connection(solenzo)
dande.add_connection(ouagadougou)
solenzo.add_connection(bobo)
solenzo.add_connection(koutiala)
solenzo.add_connection(nouna)
solenzo.add_connection(dedougou)
solenzo.add_connection(ouagadougou)
dedougou.add_connection(koutiala)
dedougou.add_connection(nouna)
dedougou.add_connection(ouagadougou)
nouna.add_connection(koutiala)
nouna.add_connection(san)
nouna.add_connection(tougan)
nouna.add_connection(ouagadougou)
tougan.add_connection(san)
tougan.add_connection(djenne)
tougan.add_connection(ouagadougou)
tougan.add_connection(mentao)
arbinda.add_connection(mentao)
arbinda.add_connection(dori)
arbinda.add_connection(goudoubo)
dori.add_connection(goudoubo)
dori.add_connection(tera)
dori.add_connection(niamey)
tera.add_connection(goudoubo)
tera.add_connection(ayorou)
tera.add_connection(niamey)
filingue.add_connection(niamey)
filingue.add_connection(mangaize)
filingue.add_connection(abala)
filingue.add_connection(goudoubo)
ayorou.add_connection(tabareybarey)
ayorou.add_connection(goudoubo)
ayorou.add_connection(mangaize)
ayorou.add_connection(niamey)
assamakka.add_connection(intikane)
assamakka.add_connection(gao)
assamakka.add_connection(kidal)
assamakka.add_connection(bordj_badji_mokhtar)
bordj_badji_mokhtar.add_connection(kidal)
bordj_badji_mokhtar.add_connection(tessalit)
bordj_badji_mokhtar.add_connection(tin_zaouaten)
tessalit.add_connection(kidal)
tessalit.add_connection(tin_zaouaten)
tin_zaouaten.add_connection(tessalit)
tin_zaouaten.add_connection(kidal)
tin_zaouaten.add_connection(assamakka)
aguelhok.add_connection(kidal)
aguelhok.add_connection(anefis)
anefis.add_connection(bourem)
anefis.add_connection(gao)
anefis.add_connection(menaka)
tintane.add_connection(tchera_rouissa)
tintane.add_connection(yelimane)
tintane.add_connection(nioro)
tintane.add_connection(timbedra)
tintane.add_connection(nema)
tintane.add_connection(walatah)
tchera_rouissa.add_connection(nioro)
tchera_rouissa.add_connection(timbedra)
tchera_rouissa.add_connection(adel_bagrou)
tchera_rouissa.add_connection(mbera)
adel_bagrou.add_connection(timbedra)
adel_bagrou.add_connection(nema)
adel_bagrou.add_connection(mbera)
adel_bagrou.add_connection(fassala)
timbedra.add_connection(nema)
timbedra.add_connection(mbera)
nema.add_connection(walatah)
nema.add_connection(mbera)
nema.add_connection(tombouctou)
walatah.add_connection(mbera)
nema.add_connection(tombouctou)

# Set up the figure for animation
fig, ax = plt.subplots(figsize=(15, 10))

# Calculate the number of days in 2012
start_date = pd.to_datetime('2012-01-01')
end_date = pd.to_datetime('2012-12-31')
dates = pd.date_range(start_date, end_date)

conflicts = pd.read_csv("Data/ACLED/1997-01-01-2024-03-01-Eastern_Africa-Middle_Africa-Northern_Africa-Southern_Africa-Western_Africa-Mali.csv")[
    pd.read_csv("Data/ACLED/1997-01-01-2024-03-01-Eastern_Africa-Middle_Africa-Northern_Africa-Southern_Africa-Western_Africa-Mali.csv")['year']==2012]

conflicts['event_date'] = pd.to_datetime(conflicts['event_date'], format="%d %B %Y")
conflicts.sort_values(by='event_date', inplace=True)

# Location dictionary required as ACLED names are not the same as OSMNX

loc_dic = {'Bamako':bamako, 
           'Kidal':kidal, 
           'Timbuktu':tombouctou, 
           'Gao':gao, 
           'Lere':lere, 
           'Menaka':menaka,
           'Syama Gold Mine':syama, 
           'Mopti':mopti,
           'Diabaly':diabaly, 
           'Koutiala':koutiala, 
           'Douentza':douentza,
           'Bourem':bourem, 
           'Sevare':sevare, 
           'Bougouni':bougouni, 
           'Ansongo':ansongo, 
           'Nara':nara, 
           'Niafunke':niafunke,
           'Aguelhok':aguelhok, 
           'Komeye Koukou':koue, 
           'Goundam':goundam, 
           'Doubabougou':bamako, 
           'Niono':niono_socoura,
           'Sari':sari, 
           'Kati':bamako, 
           'Kambila': bamako, 
           'Diago':diago, 
           'Ber':ber, 
           'Anefis':anefis, 
           'Dire':dire,
           'Tessalit':tessalit, 
           'Tenenkou':tenenkou, 
           'Imezzehene':kidal, 
           'Youwarou': youwarou, 
           'Hombori': hombori,
           'Tin Zaouaten': tin_zaouaten, 
           'Anderamboukane': anderamboukane, 
           'Imezzehene': kidal,
           'Tin Kaloumane':kidal,
           'Kayes':kayes,
           'Fana':fana,
           'Koulikoro':koulikoro,
           'Yélimané':yelimane,
           'San':san,
           'Kita':kita,
           'Sikasso':sikasso,
           'Djenne':djenne,
           'Segou':segou,
           'Nouna':nouna,
           'Mentao':mentao,
           'Dandé':dande,
           'Mbera':mbera,
           'Bobo':bobo,
           'Syama':syama,
           'Tougan':tougan,
           'Ouagadougou':ouagadougou,
           'Sélibaby':selibaby,
           'Bandiagara':bandiagara,
           'Tingrela':tingrela,
           'Banfora':banfora,
           'Dedougou':dedougou,
           'Fassala':fassala,
           'Solenzo':solenzo,
           'Arbinda':arbinda,
           'Niamey':niamey,
           'Nioro':nioro,
           'Goudoubo':goudoubo,
           'Koue':koue,
           'Tabareybarey':tabareybarey,
           'Tera':tera,
           'Mangaize':mangaize,
           'Abala':abala,
           'Ouahigouya':ouahigouya,
           'Dori':dori,
           'Ayorou':ayorou,
           'Kadiolo':kadiolo,
           'Zegoua':zegoua,
           'Intikane':intikane,
           'Filingué':filingue,
           'Bafoulabé':bafoulabe,
           'Bengassi':bengassi,
           'Kéniéba':kenieba}

# Locations without an OSMNX are assumed to be within the closest city

kidals = ['Imezzehene','Imezzehene','Tin Kaloumane']
bamakos = ['Doubabougou','Kati','Kambila']

print("Starting simulation...")
start_time = time.time()





total_population = sum(city.population for city in cities)

#########################################################################################
frac = 1000 # TO VARY
#########################################################################################

n_agents = int(total_population/frac)

for loc in locations:
    loc.population= int(loc.population/frac)

for camp in camps:
    camp.capacity = int(camp.capacity/frac)

city_probabilities = {city.name: city.population / total_population for city in cities}

prob_values = list(city_probabilities.values())

# Normalize the probabilities
normalized_prob_values = [float(i)/sum(prob_values) for i in prob_values]

if not np.isclose(sum(normalized_prob_values), 1.0):
    raise ValueError("Normalized probabilities do not sum closely to 1.")

Agents = {}
for i in list(range(1,n_agents+1)):
    Agents[i] = Agent(i)

G = create_graph(locations)
ongoing_conflicts = []

Agent.calculate_distributions()
Agent.initialise_cities(foreign_cities)

populations = {camp.name: [] for camp in camps}

total_agents = len(Agents)

for current_date in dates: 

    """
    
    This is where the model will be ran in real time
    
    """   
    processed_agents = 0

    print("\n")
    print(f"Simulating day: {current_date.strftime('%Y-%m-%d')}")

    for ongoing_conflict in ongoing_conflicts:
        ongoing_conflict.check_and_update_conflict_status(current_date) # check ongoing conflicts
        if ongoing_conflict.hasconflict:
            pass
        else:
            ongoing_conflicts.remove(ongoing_conflict) # remove conflict from stored list
            G.nodes[ongoing_conflict.name]['has_conflict']=False # update graph node

    for idx, event in conflicts[conflicts['event_date'] == current_date].iterrows(): # consider all conflicts in given day

        location = loc_dic[event['location']]

        # print("Conflict in " + event['location'])

        # Dealing with the exceptions between ACLED and OSMNX
        if event['location'] in kidals:
            G.nodes['Kidal']['has_conflict']=True
        elif event['location'] in bamakos:
            G.nodes['Bamako']['has_conflict']=True
        elif event['location']=='Tin Zaouaten':
            G.nodes['Tinzaouaten']['has_conflict']=True
        elif event['location']=='Komeye Koukou':
            G.nodes['Koue']['has_conflict']=True
        elif event['location']=='Syama Gold Mine':
            G.nodes['Syama']['has_conflict']=True
        else:
            G.nodes[event['location']]['has_conflict']=True
        location.in_city_conflict(event['fatalities'], current_date)
        if location not in ongoing_conflicts:
            ongoing_conflicts.append(location)

    # represent each epoch as a graph, commented out for now due to large number of unneccesary graphs.
    # draw_graph(G, current_date,ongoing_conflicts)

    # print(colors.PURPLE, "current conflicts",[con.name for con in ongoing_conflicts], colors.END)
    
    for id in Agents:
        Agents[id].assess_situation_and_move_if_needed(loc_dic[Agents[id].location],current_date)
        loc_dic[Agents[id].location].population -= 1 # update locations
        loc_dic[Agents[id].shortterm].population += 1
        G.nodes[Agents[id].location]['population'] -= 1 # update nodes
        G.nodes[Agents[id].shortterm]['population'] += 1

        processed_agents += 1  # Update the counter after processing each agent
        print_progress_bar(processed_agents, total_agents, prefix='Progress:', suffix='Complete', length=50)

    for camp in camps:
        populations[camp.name].append(camp.population)
    
camp_names = list(populations.keys())
n_camps = len(camp_names)
camps_per_figure = 3

# Loop through the camps in chunks of 3
for i in range(0, n_camps, camps_per_figure):
    fig, axes = plt.subplots(nrows=1, ncols=camps_per_figure, figsize=(15, 5))
    
    for j in range(camps_per_figure):
        camp_index = i + j
        if camp_index < n_camps:  # Check to avoid index out of range
            ax = axes[j]
            camp_name = camp_names[camp_index]
            ax.plot(dates, populations[camp_name], label=camp_name)
            ax.set_title(camp_name)
            ax.set_xlabel('Date')
            ax.set_ylabel('Population')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
        else:
            axes[j].axis('off')  # Hide unused subplot
    
    plt.tight_layout()
    plt.show()

csv_file = 'Simulated_pop_data.csv'

populations['Date']=dates

# Write the dictionary to a CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=populations.keys())
    
    # Write header
    writer.writeheader()
    
    # Write data rows
    for i in range(len(populations['Date'])):
        row = {key: populations[key][i] for key in populations.keys()}
        writer.writerow(row)

end_time = time.time()


elapsed_time = end_time - start_time

print(f"Simulation completed in {elapsed_time:.2f} seconds.")

"""
# e.g. where can an agent access within a day, starting at Bamako and travelling at 200km/d
accessible = find_accessible_nodes_within_distance(G, 'Bamako', 200)

print('Walking from Bamako:',accessible)

# e.g. how do I get to all camps from Bamako (shortest distance) (max move speed of 200)

camp_paths_from_bamako=camp_paths(G, 'Bamako',200)


if camp_paths_from_bamako is not None:
    for key in camp_paths_from_bamako:
        print('To get to %s from Bamako it takes %.1fkm and you travel through: %s' % (key, camp_paths_from_bamako[key]['distance'], camp_paths_from_bamako[key]['path']))


# e.g. how do I get to the nearest airport from Zegoua (max move speed of 200)
    
air_path = find_nearest_city_with_airport_directly(G, 'Bamako',200) 

print(air_path)

# How do I get to a bordering country e.g. Senegal (max move speed of 200)

count_path = find_shortest_paths_to_neighboring_countries(G, 'Bamako',200)

if count_path is not None:
    print(count_path)
"""

for camp in camps:
    print("Camp in " + str(camp.name) + " is " + str(camp.population))


"""
for id in Agents:
    if Agents[id].capitalbracket == 'Rich':
        print(colors.GREEN + Agents[id].location + colors.END)
    elif Agents[id].capitalbracket == 'Mid':
        print(colors.YELLOW + Agents[id].location + colors.END)
    else:
        print(colors.RED + Agents[id].location + colors.END)
"""
# draw_graph(G, current_date, distances_on=True)
        