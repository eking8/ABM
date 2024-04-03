import numpy as np
import random
import networkx as nx
from datetime import datetime


class Agent:
    # Age groups and their population
    age_groups = [(90, 100), (80, 89), (70, 79), (60, 69), (50, 59), (40, 49), (30, 39), (20, 29), (10, 19), (0, 9)]
    populations = [21510, 88052, 289198, 632542, 984684, 1569836, 2451989, 3466640, 5240091, 7650947]

    probabilities = []
    cumulative_distribution = []

    @classmethod
    def initialise_populations(cls,cities,total_population):

        cls.city_probabilities = {city.name: city.population / total_population for city in cities}
        prob_values = list(cls.city_probabilities.values())

        # Normalize the probabilities
        cls.normalized_prob_values = [float(i)/sum(prob_values) for i in prob_values]

        if not np.isclose(sum(cls.normalized_prob_values), 1.0):
            raise ValueError("Normalized probabilities do not sum closely to 1.")


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
        self.longterm = None  # Intended destination
        self.moving = False  # Is the agent currently moving to a destination
        self.threshold = np.random.uniform(0,30)  # InitiaClise the threshold here or in a separate method
        rand_n= np.random.uniform(0,1)
        self.startdate=None
        self.enddate=None
        self.traveltime=None
        self.nogos=[]
        self.country_origin='Mali'
        self.is_stuck=False
        self.been_abroad=False
        self.is_leader=False 
        self.in_family=False # Replace with statistical distribution
        self.fam_size=1
        self.fam=None
        self.is_leader=True
        self.location = np.random.choice(
            list(self.__class__.city_probabilities.keys()),
            p=self.__class__.normalized_prob_values
        )
        self.city_origin = self.location
        self.shortterm = self.location

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

    def assess_situation_and_move_if_needed(self,G,city,current_date):
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