import numpy as np
import random
import networkx as nx
from datetime import datetime
from Visualisations import colors
import math as math


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

    def __init__(self, id, status = 'Resident', age=None, is_leader=False):
        self.id = id

        if age:
            self.age = age
        else:
            self.age=self.generate_random_age()
        
        self.gender = self.generate_gender()
        self.status = status
        self.longterm = None  # Intended destination
        self.moving = False  # Is the agent currently moving to a destination
        self.threshold = np.random.uniform(0,30)  # InitiaClise the threshold here or in a separate method
        rand_n= np.random.uniform(0,1)
        self.startdate=None
        self.enddate=None
        self.traveltime=None
        self.nogos=set()
        self.country_origin='Mali'
        self.is_stuck=False
        self.been_abroad=False
        self.is_leader=is_leader
        self.in_family=False 
        self.fam_size=1
        self.fam=None
        self.location = np.random.choice(
            list(self.__class__.city_probabilities.keys()),
            p=self.__class__.normalized_prob_values
        )
        self.city_origin = self.location
        self.shortterm = self.location
        self.fam=[self] # carried by all members of the family
        self.merged=False
        self.travelswithfam=False
        self.group=[self] # carried by leader only
        self.ingroup=False
        self.checked=False
        self.leftfam=False
        self.familiar={}
        self.distance_traveled_since_rest=0
        

        if rand_n>0.9965: # capital 100000
            self.capitalbracket = 'Rich'
        elif rand_n>0.6094: # capital > 10000
            self.capitalbracket = 'Mid'
        else:
            self.capitalbracket = 'Poor'

        if self.capitalbracket == 'Poor': # No access to car
            self.speed = abs(np.random.normal(200,50))
        else:
            self.speed = abs(np.random.normal(400, 100))
    

    def form_families(self,all_agents):     


        if not self.in_family and not self.checked:

            if 16<self.age<65:
                

                self.in_family=True
                
                local_agents = [agent for agent in all_agents if agent.location == self.location and agent.id != self.id]
                available_agents = [agent for agent in local_agents if not agent.in_family and not agent.checked and agent.id>self.id]
                
                random_poisson = np.random.poisson(6.18)
                fam_size=int(random_poisson + np.random.normal(0,0.1))

                if fam_size < 1:
                    fam_size = 1
                    self.in_family=False
                
                self.fam_size = fam_size

                self.fam += random.sample(available_agents, min(self.fam_size -1, len(available_agents)))
                    
                
                for agent in self.fam:
                    agent.in_family=True
                    agent.fam=self.fam
                    agent.checked=True

                self.is_leader=True
                
        

            
                
    def form_fam_group(self): # needs to account for bayesian likelihood of travelling with your fam
        # prob of travelling with fam / prob of being in fam = prob of travelling with fam given in fam
        if self.in_family:
            self.travelswithfam= random.random() < 0.596/0.98 
        else:
            self.is_leader=True




    def define_groups(self):
        if self.in_family and not self.ingroup:
            
            if self.is_leader:
                
                if self.travelswithfam:
                    
                    
                    self.group = [x for x in self.fam if x.travelswithfam]
                    self.group= sorted(self.group, key=lambda agent: agent.id)
                    
                    if len(self.group)>1:
                        for agent in self.group:
                            agent.group=self.group
                            agent.ingroup=True
                    
                else:
                    self.leftfam=True
                    max_member = max((x for x in self.fam if x.travelswithfam and x.age < 65), key=lambda x: x.age, default=None)
                    if max_member:
                        max_member.is_leader=True
                        max_member.group=[x for x in max_member.fam if x.travelswithfam]
                        max_member.group = sorted(max_member.group, key=lambda agent: agent.id)
                        if len(max_member.group)>1:
                            max_member.ingroup=True
                            for agent in max_member.group:
                                agent.group=max_member.group
                                agent.ingroup=True

            
            else:
                if not self.travelswithfam:
                    self.leftfam=True
                    self.is_leader=True
        
        
                    

        
            # strategic group stuff here

        
            


            

    def group_speeds_and_cap(self):
        
        self.speed = min(x.speed for x in self.group)

        if self.is_leader:
            for agent in self.group:
                agent.capitalbracket=self.capitalbracket

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
    
    def kill(self, frac):
        val = random.randint(1, frac)
        if val==1:
            self.status='Dead'
        
            if self.is_leader and self.ingroup:
                max_member = max((x for x in self.fam if x.travelswithfam and x.age < 65), key=lambda x: x.age, default=None)
                if max_member:
                    max_member.is_leader=True    
            
            agents = self.group
            for agent in agents:
                if self in agent.group:
                    agent.group.remove(self)

        


    def assess_situation_and_move_if_needed(self,G,city,current_date):
        
        
        if self.is_leader and not self.is_stuck and self.location!='Abroad' and self.status != 'Dead':
            
            rest_prob = 1-math.exp(-self.distance_traveled*(current_date-city.last_conflict_date)/36500)
            
            if random.random()>rest_prob:
                self.merged = False

                
                if city.hasconflict and city.fatalities > self.threshold:
                    for agent in self.group:
                        agent.nogos.update(city.name)
                        if agent.status == 'Resident':
                                    
                                agent.moving = True
                                agent.status = 'Fleeing from conflict'
                                # print(colors.RED + "Agent " + str(self.id) + " is now fleeing from " + str(self.location) + colors.END)
                                agent.startdate=current_date
                                
                                    


                if self.moving and self.status in ['Refugee','Returnee','IDP','Fleeing from conflict']:

                    
                    for agent in self.group:
                        agent.distance_traveled_since_rest+=G[agent.location][agent.shortterm]['distance']
                        agent.location=agent.shortterm
                        if not agent.moving:
                            agent.moving=True
                            agent.startdate=current_date

                    if self.location in self.__class__.foreign_cities:
                        new_status = 'Refugee'
                        self.been_abroad = True
                    elif self.been_abroad:
                        new_status = 'Returnee'
                    else:
                        new_status = 'IDP'

                    for agent in self.group:
                        agent.status = new_status
                        agent.been_abroad = self.been_abroad
                        agent.traveltime = current_date - agent.startdate

                        if self.location == self.longterm:
                            agent.moving = False
                            agent.enddate = current_date
                            
                            if self.capitalbracket == 'Rich':
                                agent.location = 'Abroad'
                                agent.been_abroad = True

                            # print(colors.GREEN + "Agent " + str(self.id) + " has reached " + str(self.longterm) + colors.END)
                            
                            # LOGIC THAT ASSIGNS STATUS AND MOVING CHANGE FOR FAMILY (FOLLOWERS)
                        
                
                    
                    destination_dict = None

                    if self.capitalbracket == 'Rich':
                        destination_dict = find_nearest_cities_with_airport(G, self.location, self.speed, self.nogos)
                        iscamp=False
                    elif self.capitalbracket == 'Mid':
                        destination_dict = find_shortest_paths_to_neighboring_countries(G, self.location, self.speed, current_date, self.nogos)
                        iscamp=False
                    else:
                        destination_dict = camp_paths(G, self.location, self.speed, current_date, self.nogos)
                        iscamp=True

                    if destination_dict:
                        key = self.roulette_select(destination_dict,iscamp)
                        
                        if key:
                            for agent in self.group:
                                agent.distanceleft = destination_dict[key]['distance']
                                agent.longterm = key
                                agent.shortterm = destination_dict[key]['path'][0]
                    
                                # Debugging print statements can be uncommented for additional logs
                                print("location = " + self.location + ", and short term = " + str(self.shortterm))
                                print(colors.YELLOW + "Agent " + str(self.id) + " is going to the camp in " + str(self.longterm) + " from "+ str(self.location) + colors.END)
                    else:
                        for agent in self.group:
                            agent.is_stuck = True
                            agent.moving = False


            else:
                self.distance_traveled_since_rest=0 # reset after rest
                    
    
    def merge_nogo_lists(self, all_agents):
        """
        This method allows an agent to merge their 'nogos' set with those of 0-3 other agents in the same city.
        :param all_agents: List of all Agent instances
        """
        # Filter agents in the same city and not the same agent
        local_agents = [agent for agent in all_agents if agent.location == self.location and agent.id != self.id and not agent.merged]

        # Randomly select 1-3 agents to interact with
        number_of_agents = random.randint(0, 3)
        selected_agents = random.sample(local_agents, min(number_of_agents, len(local_agents)))

        if number_of_agents>0:
            self.merged=True

        # Merge the nogo lists
        for agent in selected_agents:
            if agent in self.fam:
                agent.merged=True
                # Update each agent's nogo list
                combined_nogos = self.nogos.union(agent.nogos)
                self.nogos = agent.nogos = combined_nogos
                # familiar destinations passed on
                self.familiar[agent.longterm] = self.familiar.get(agent.longterm, 0) + 1
                agent.familiar[self.longterm] = agent.familiar.get(self.longterm, 0) + 1
                

                
        
        


    
    def roulette_select(self, distances,iscamp):
        """
        Select a key from the distances dictionary using roulette method,
        where scores are computed as `familiar - a * distance + b - c * population`.

        :param distances: Dictionary of {key: {'distance': value}}
        :return: Selected key based on roulette selection
        """

        if distances is None:
            print('No routes')
            return None

        # Filter out routes with distance of 0, as these can't be scored properly
        distances = {key: value for key, value in distances.items() if value['distance'] != 0}

        a = self.age/1000 
        b = 50 # bias assumed to be 50 

        if iscamp:
            c = 1/2000
        else:
            c=0
        
        # Compute scores using familiarity and distance
        scores = {key: self.familiar.get(key, 0) - a * distances[key]['distance'] + b - c*distances[key]['population'] for key in distances}
        
        # Calculate total sum of scores (only positive scores contribute to the roulette wheel)
        total_score_sum = sum(max(score, 0) for score in scores.values())

        if total_score_sum == 0:
            return None
        
        # Normalize scores to probabilities
        probabilities = {key: max(scores[key], 0) / total_score_sum for key in scores}
        
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
    
    pops={}
    
    distances, paths = nx.single_source_dijkstra(G_filtered, start_node)
    camp_paths_distances = {}
    for node, data in G.nodes(data=True):
        if G.nodes[node].get('type') == 'Camp' and node in paths:
            if data.get('country',None) in ['Niger','Burkina Faso'] and current_date < datetime(2012, 2, 21).date():
                pass # boarder does not open before 21st feb
            else:
                pops[node]=data.get('population',0)
                paths[node].remove(start_node)
                camp_paths_distances[node] = {'path':paths[node], 'distance':distances[node],'population':pops[node]}

    
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
    if start_node in nogo:
        nogo.remove(start_node)

    G_filtered = filter_graph_by_max_link_length(G, max_link_length,start_node,nogo=nogo)

    if not G_filtered.has_node(start_node):
        print('Error: Start node is not present in the filtered graph.')
        return {}
    
    pops={}
    
    for node, data in G.nodes(data=True):
        pops[node]=data.get('population',0)

    cities_info = {}

    for node in G_filtered.nodes:
        if node != start_node and G_filtered.nodes[node].get('has_airport', False):
            try:
                path_length = nx.shortest_path_length(G_filtered, start_node, node, weight='weight')
                path = nx.shortest_path(G_filtered, start_node, node, weight='weight')[1:]  # Exclude start node from path
                cities_info[node] = {'distance': path_length, 'path': path, 'population':pops[node]}
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

    pops={}
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
                path = nx.shortest_path(G_filtered, start_node, node, weight='weight')[1:]
                pops[node]=data.get('population',0)
                # If the city is closer than any previously found or if it's the first city found for this country
                if node not in results or path_length < results[node]['distance']:
                    results[node] = {'distance': path_length, 'path': path, 'population':pops[node]}
            except nx.NetworkXNoPath:
                pass  # No path exists to this node, ignore it

    return results


def filter_graph_by_max_link_length(G, max_link_length, start_node, nogo= []):
    # Step 1: Create a new empty graph to hold the filtered graph
    G_filtered = nx.Graph()

    
    # Step 2: Copy all nodes from G to G_filtered, preserving attributes
    for node, data in G.nodes(data=True):
        G_filtered.add_node(node, **data)
            
    if start_node not in G_filtered:
        G_filtered.add_node(start_node)
    
    # Step 3: Filter edges by max_link_length and add them to G_filtered
    for u, v, d in G.edges(data=True):
        if d.get('weight', float('inf')) <= max_link_length:
            G_filtered.add_edge(u, v, **d)
    
    return G_filtered

def deathmech(members,fat):
    if fat==0:
        return None
    elif len(members)>fat:
        death_ids=random.sample(members,fat)
    else:
        death_ids=members
    
    return members