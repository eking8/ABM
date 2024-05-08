import numpy as np
import random
import networkx as nx
from datetime import datetime
from Visualisations import colors
import math as math
from datetime import timedelta
import pandas as pd
import sys


class Agent:
    # Age groups and their population
    age_groups = [(90, 100), (80, 89), (70, 79), (60, 69), (50, 59), (40, 49), (30, 39), (20, 29), (10, 19), (0, 9)]
    populations = [21510, 88052, 289198, 632542, 984684, 1569836, 2451989, 3466640, 5240091, 7650947]
    lam = 0.225

    probabilities = []
    cumulative_distribution = []


    @classmethod
    def initialise_populations(cls,locations,total_population):

        cls.city_probabilities = {location.name: location.population/ total_population for location in locations}
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
    

    @classmethod
    def randomloc(cls):
        val = np.random.choice(
                list(__class__.city_probabilities.keys()),
                p=__class__.normalized_prob_values
            )
        return val
    

    def __init__(self, id, status = 'Resident', age=None, is_leader=False,location=None):
        self.id = id

        if age:
            self.age = age
        else:
            self.age=self.generate_random_age()
        
        self.gender = self.generate_gender()
        self.status = status
        self.longterm = None  # Intended destination
        self.moving = False  # Is the agent currently moving to a destination
        self.threshold = np.random.uniform(0,20) 
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
        if location:
            self.location=location
        else:
            self.location = np.random.choice(
                list(self.__class__.city_probabilities.keys()),
                p=self.__class__.normalized_prob_values
            )
        self.city_origin = self.location
        self.shortterm = self.location
        self.fam=[self] 
        self.merged=False
        self.travelswithfam=False
        self.group=[self] 
        self.ingroup=False
        self.checked=False
        self.leftfam=False
        self.familiar={}
        self.distance_traveled_since_rest=0
        self.moved_today=False
        self.direction=None
        self.instratgroup=False
        self.comb=False
        rand_n2=np.random.uniform(0,1)
        self.media= rand_n2<=0.06
        self.contacts=[]
        self.contacts_in_camp={}
        self.tellsfam=0.06<=rand_n2<=0.56 # 50% get info from fam
        

        if rand_n>0.9965: # capital 100000
            self.capitalbracket = 'Rich'
        elif rand_n>0.6094: # capital > 10000
            self.capitalbracket = 'Mid'
        else:
            self.capitalbracket = 'Poor'

        if self.capitalbracket == 'Poor': # No access to car
            self.speed = abs(np.random.normal(160,50))+44.2
            self.origspeed=self.speed
        else:
            self.speed = abs(np.random.normal(500, 100))
            self.origspeed=self.speed

    
    def form_families(self,all_agents):     


        if not self.in_family and not self.checked:

            if 16<=self.age<=65:
                

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
            

    def join_dependents(self,all_agents):
        if not self.in_family and not self.is_leader:
            if self.age>65 or self.age<16:
                available_leaders = [agent for agent in all_agents if agent.location == self.location 
                                     and agent.id != self.id and agent.is_leader]
                if len(available_leaders)>0:
                    leader = random.choice(available_leaders)
                
                    self.in_family=True

                    if leader.in_family:
                        leader.fam.append(self)
                        for agent in leader.fam:
                            agent.fam=leader.fam
                
                    else:
                        leader.in_family=True
                        leader.fam.append(self)
                        self.fam.append(leader)
                else:
                    self.is_leader # forced to travel alone without family            
          
    def form_fam_group(self): # Bayesian probability of travelling with fam given that they are an independent age and in a family
       
        if 16<self.age<65:
            if self.in_family:
                self.travelswithfam = random.random()<0.1761
            else:
                self.is_leader=True
        else:
            if self.is_leader:
                self.travelswithfam=False
            else:
                self.travelswithfam=True
        self.checked=False


    def define_groups(self,all_agents):
        if self.in_family and not self.ingroup and not self.checked:
            
            if self.is_leader:
                
                if self.travelswithfam:
                                        
                    self.group = [x for x in self.fam if x.travelswithfam]
                    self.group= sorted(self.group, key=lambda agent: agent.id)
                    
                    if len(self.group)>1:
                        for agent in self.group:
                            agent.group=self.group
                            agent.ingroup=True
                            agent.checked=True
                    else:
                        self.checked=True
                        self.ingroup=False
                        self.leftfam=True
                        if self.tellsfam:
                            self.contacts+=[x for x in self.fam if x not in self.contacts]
                        
                    
                else:
                    self.leftfam=True
                    if self.tellsfam:
                        self.contacts+=[x for x in self.fam if x not in self.contacts]
                    max_member = max((x for x in self.fam if x.travelswithfam and 16<x.age < 65), 
                                     key=lambda x: x.age, default=None)
                    if max_member:
                        max_member.is_leader=True
                        max_member.group=[x for x in max_member.fam if x.travelswithfam]
                        max_member.group = sorted(max_member.group, key=lambda agent: agent.id)
                        if len(max_member.group)>1:
                            
                            for agent in max_member.group:
                                agent.group=max_member.group
                                agent.ingroup=True
                                agent.checked=True
                            else:
                                max_member.checked=True
                                max_member.ingroup=False
                                max_member.leftfam=True
                                if max_member.tellsfam:
                                    max_member.contacts+=[x for x in max_member.fam if x not in max_member.contacts]
                                
                    else:
                        for member in self.fam:
                            if not member.checked:
                                member.joingroup(all_agents)
                                
            
            else:
                if not self.travelswithfam:
                    self.leftfam=True
                    self.is_leader=True
                    self.checked=True   
                    if self.tellsfam:
                        self.contacts+=[x for x in self.fam if x not in self.contacts]                        

    def joingroup(self,all_agents,nolead=None):

        available_leaders = [agent for agent in all_agents if agent.location == self.location 
                             and agent.id != self.id and agent.is_leader and 16 <= agent.age <= 65 
                             and agent != nolead]
        
        if len(available_leaders)>0:
            leader = random.choice(available_leaders)
            self.ingroup=True
            self.is_leader=False

            rejoin=False
            self.checked=True
            if leader.ingroup:
                for agent in leader.fam:
                    agent.group.append(self)
                    self.group.append(agent)
                    agent.checked=True
                    if agent in self.fam and agent.leftfam:
                        agent.leftfam=False
                        if agent.tellsfam:
                            agent.contacts+=[x for x in agent.fam if x not in agent.contacts]
                        rejoin=True
                        
                if rejoin:
                    self.leftfam=False
                              
            else: 
                leader.ingroup=True
                leader.group.append(self)
                self.group.append(leader)
                
                if leader in self.fam and leader.leftfam:
                    leader.leftfam=False
                    self.leftfam=False
        else:
            self.is_leader=True # Forced to travel alone
            if self.in_family:
                self.leftfam=True     
                if self.tellsfam:
                        self.contacts+=[x for x in self.fam if x not in self.contacts]          

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
        
        # Efficiently find the index using searchsorted
        index = np.searchsorted(self.__class__.cumulative_distribution, rand)
        
        # Get the corresponding age group based on the index found
        age_min, age_max = self.__class__.age_groups[index]

        # Return a random age within the selected age group
        return np.random.randint(age_min, age_max + 1)
    
    @staticmethod
    def generate_gender():
        probabilities = [0.503, 0.497]
        choices = ['M', 'F']
        return random.choices(choices, weights=probabilities, k=1)[0]
    
    def kill(self, frac, all_agents):
        # Determine if the agent will die based on random chance
        if random.randint(1, frac) == 1:
            self.status = 'Dead'
            self.location = 'Dead'
            self.instratgroup = False

            # Handle group-related logic only if the agent is part of a group
            if self.ingroup:
                # Define variables to determine the new leader and speed adjustment
                new_leader = None
                min_speed = float('inf')

                # Iterate once through the group members
                for member in self.group:
                    if member.status != 'Dead' and member.age < 65:
                        # Determine potential new leader
                        if new_leader is None or member.age > new_leader.age:
                            new_leader = member
                    # Calculate the minimum speed of alive agents excluding self
                    if member != self:
                        min_speed = min(min_speed, member.origspeed)
                    # Remove self from all member's groups
                    if self in member.group:
                        member.group.remove(self)

                # Assign new leader if available
                if new_leader:
                    new_leader.is_leader = True
                else:
                    # If no new leader, all members join other groups
                    for member in self.group:
                        member.speed=member.origspeed
                        member.joingroup(all_agents)

                
                for member in self.group:
                    member.speed = min_speed
    
    def assess_situation_and_move_if_needed(self,G,city,current_date,roulette=True):
    
        if self.ingroup and len(self.group)==1:
            self.ingroup=False
                     
        if self.is_leader and not self.is_stuck and self.location!='Abroad' and self.status != 'Dead':
            
            self.merged = False
                
            if city.hasconflict and city.fatalities > self.threshold:
                for agent in self.group:
                    agent.nogos.add(city.name)
                    if agent.status == 'Resident':
                            
                            agent.moving = True
                            agent.status = 'Fleeing from conflict'
                            agent.startdate=current_date
                            agent.distance_traveled_since_rest=0                                    

            if (self.moving and self.status in ['Refugee','Returnee','IDP','Fleeing from conflict']) or (city.name=='Fassala' and current_date >= pd.Timestamp(datetime(2012, 3, 19).date())):
                
                if self.location in self.__class__.foreign_cities:
                    new_status = 'Refugee'
                    self.been_abroad = True
                elif self.been_abroad:
                    new_status = 'Returnee'
                else:
                    new_status = 'IDP'

                if city.last_conflict_date:
                    cooldown=(current_date-city.last_conflict_date).days
                    add = city.waited_days/1000
                    cooldown*=(1+add)
                else:
                    cooldown=22 # this cool down ensures 1/3 chance of moving on 100km walked

                rest_prob = 1-math.exp(-(self.distance_traveled_since_rest+100)*cooldown/3300) 
                # at 14 days after conflict, 200km walked, and 10 people stopping should be 1/3 chance of moving

                if random.random()>rest_prob or (city.name=='Fassala' and current_date >= pd.Timestamp(datetime(2012, 3, 19).date())):

                    self.moved_today=True
                    
                    for agent in self.group:
                        if agent.shortterm not in ['Dead',None]:
                            agent.distance_traveled_since_rest+=nx.shortest_path_length(G, agent.location, agent.shortterm, weight='weight')
                            agent.location=agent.shortterm
                        if not agent.moving:
                            agent.moving=True
                            agent.startdate=current_date

                    

                    for agent in self.group:
                        agent.status = new_status
                        agent.been_abroad = self.been_abroad
                        agent.traveltime = current_date - agent.startdate

                        if self.location == self.longterm:
                            if current_date >= pd.Timestamp(datetime(2012, 3, 19).date()) and self.location=='Fassala':
                                pass
                            else:
                                agent.moving = False
                                agent.enddate = current_date
                                
                                if self.capitalbracket == 'Rich':
                                    agent.location = 'Abroad'
                                    agent.been_abroad = True
                                    self.status='International Refugee'
                            
                                for agent2 in self.contacts:
                                    if agent.location in agent2.contacts_in_camp:
                                        agent2.contacts_in_camp[agent.location]+=1
                                    else:
                                        agent2.contacts_in_camp[agent.location]=1

                                

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
                        if roulette:
                            key = self.roulette_select(G,city.name,destination_dict,iscamp)
                        else:
                            key = min(destination_dict, key=lambda k: destination_dict[k]['distance'])

                        if key:
                            for agent in self.group:
                                agent.distanceleft = destination_dict[key]['distance']
                                agent.longterm = key
                                agent.shortterm = destination_dict[key]['path'][0]
                    
                    else:
                        for agent in self.group:
                            agent.is_stuck = True
                            agent.moving = False
                            agent.moved_today= False
                            
                else:
                    self.distance_traveled_since_rest=0 # reset after rest
                    city.waited_days+=len(self.group)

        elif self.is_stuck or (self.location=="Fassala" and current_date>pd.Timestamp(datetime(2012, 3, 19))):
            if city:
                G_filtered = filter_graph_by_max_link_length(G, self.speed,city,nogo=self.nogos)
                if set(G_filtered.neighbors(city.name)).issubset(self.nogos):
                    for neighbor in G_filtered.neighbors(city.name):
                        self.nogos.remove(neighbor)

    def merge_nogo_lists(self, all_agents):
        """
        This method allows an agent to merge their 'nogos' set with those of 0-3 other agents in the same city.
        :param all_agents: List of all Agent instances
        """
        # Filter agents in the same city and not the same agent
        local_agents = [agent for agent in all_agents if agent.location == self.location and agent.id != self.id and not agent.merged and agent not in self.group and agent.status!='Dead' and agent.is_leader]

        # Randomly select 1-3 agents to interact with
        number_of_agents = random.randint(0, 3)
        selected_agents = random.sample(local_agents, min(number_of_agents, len(local_agents)))

        if number_of_agents>0:
            self.merged=True

        # Merge the nogo lists
        for agent in selected_agents:

            if agent.media and self.media:
                agent.contacts.append(self)
                self.contacts.append(agent)

            agent.merged=True
            # Update each agent's nogo list
            combined_nogos = self.nogos.union(agent.nogos)
            self.nogos = agent.nogos = combined_nogos
            # familiar destinations passed on
            self.familiar[agent.longterm] = self.familiar.get(agent.longterm, 0) + 1
            agent.familiar[self.longterm] = agent.familiar.get(self.longterm, 0) + 1

            for agent2 in agent.group:
                agent2.nogos=agent.nogos
                agent2.familiar=agent.familiar


        for agent in self.group:
            agent.nogos=self.nogos
            agent.familiar=self.familiar
                

    def roulette_select(self, G, startnode, distances, iscamp):
        """
        Select a key from the distances dictionary using roulette method,
        where scores are computed as `familiar - a * distance + b - c * population + d * cos(current direction - new direction) - e*fatalities
        + f*number travelled on first link + g*number of contacts made to camp'

        The agents have a utility to:
        - Familiarity
        - Directions similar to the current one (more likely to stick out for the long term)
        - Number travelled on first link (SOCIAL NETWORK THEORY)
        - Number of contacts made to camp
        
        The agents have a disutility to:
        - Large distances
        - Overpopulated camps
        - Directions opposite to current one (less likely to circle around)
        - Fatalities on nodes

        :param distances: Dictionary of {key: {'distance': value, 'population': value, 'path': [path]}}
        :return: Selected key based on roulette selection
        """

        if not distances:
            print('No routes')
            return None

        # Define constants and parameters
        a = self.age / 1000
        b = 50  # Constant bias
        c = 1/200 if iscamp else 0
        d = 50  # Cosine multiplication factor
        e = 10
        f = 1
        g = 1

        scores = {}
        total_score_sum = 0

        for key, value in distances.items():
            if value['distance'] == 0:
                continue  # Skip zero distance

            # Calculate bearing difference and its cosine
            bearing_difference = 0
            edge_data = G.get_edge_data(startnode, value['path'][0], default={})
            if self.direction and 'bearing' in edge_data:
                bearing_difference = self.direction - edge_data['bearing']
            
            cosine_of_bearing = math.cos(bearing_difference / 2)

            # Calculate the score using the given formula
            score = (self.familiar.get(key, 0)
                    - a * value['distance']
                    + b
                    - c * value.get('population', 0)
                    + d * cosine_of_bearing
                    - e * G.nodes[key].get('fatalities', 0)
                    + f * value.get('travelled', 0)
                    + g * self.contacts_in_camp.get(key, 0))

            if score > 0:
                scores[key] = score
                total_score_sum += score

        if total_score_sum == 0:
            return None

        # Roulette wheel selection
        r = random.random()
        cumulative_prob = 0.0

        for key, score in scores.items():
            cumulative_prob += score / total_score_sum
            if r <= cumulative_prob:
                return key

        # If no selection made due to rounding errors, return the last key considered
        return key
    
    def indirect_check(self, G, start_node,current_date):

        G_filtered = filter_graph_by_max_link_length(G, 100,start_node,self.nogos)
        
        if G_filtered.nodes[start_node].get('is_camp',False) and (current_date<pd.Timestamp(datetime(2012, 3, 19)) or start_node!='Mbera'):


            for neighbor in G_filtered.neighbors(start_node):
                
                if G_filtered.nodes[neighbor].get('has_conflict', False):
                    if neighbor=='Fassala' and current_date>pd.Timestamp(datetime(2012, 3, 19)):
                        for agent in self.group:
                            agent.nogos.add(G_filtered.nodes[neighbor].get('name', neighbor))
                            agent.nogos.add(G_filtered.nodes[start_node].get('name', start_node))
                            agent.moving=True
                            agent.startdate=current_date
                        break

        else:

            for neighbor in G_filtered.neighbors(start_node):
                
                if G_filtered.nodes[neighbor].get('has_conflict', False):
                    for agent in self.group:
                        agent.nogos.add(G_filtered.nodes[neighbor].get('name', neighbor))
    

    def check_kick_out(self,ags):
        N=len(self.group)
        if N>1:
            if self.origspeed==self.speed:
                # group bottleneck
                second_slowest=min([x.origspeed for x in self.group if x!=self])
                speed_norm=self.speed/second_slowest
                grp_ut=self.lam*speed_norm+(1-self.lam)*(N-0.5)/N
                w0_ut=self.lam*1+(1-self.lam)*(N-1.5)/N
            
                if w0_ut>grp_ut:
                        self.kickout(ags)
        else:
            self.ingroup=False
            self.instratgroup=False
            self.is_leader=True
    

    def kickout(self,ags):
        if self.is_leader:
            max_member = max((x for x in self.group if x.status!='Dead' and x.age < 65), key=lambda x: x.age, default=None)
            if max_member:
                
                max_member.is_leader=True   
                minspeed=min([x.origspeed for x in self.group])
                for agent in self.group:
                    if self in agent.group and agent != self:
                        agent.group.remove(self)
                        agent.speed=minspeed

                self.group=[self]
                lead=max_member

            else:
                for member in self.group:
                    member.ingroup=False
                    member.instratgroup=False
                    member.group=[member]
                    member.speed=member.origspeed
                    member.joingroup(ags) 
                lead=None
                    
        else:
            
            minspeed=min([x.speed for x in self.group if x != self])
            for agent in self.group:
                if agent != self:
                    print(agent.group)
                    agent.group.remove(self)
                    agent.speed=minspeed
                    if agent.is_leader:
                        lead=agent
                
        self.is_leader=True
        self.ingroup=False
        self.instratgroup=False
        self.group=[self]
        self.speed=self.origspeed
        
        if not 16<self.age<65:
                self.joingroup(ags,nolead=lead)
                


    def super_imp(self,other):

        if other in self.group:
            print('There must not be 2 leaders in same group')
        
        N1=len(self.group)
        N2=len(other.group)
        if self.speed>other.speed:
            speed_norm_1=1
            speed_norm_2=other.speed/self.speed
            speed_norm_12=speed_norm_2
        else:
            speed_norm_1=self.speed/other.speed
            speed_norm_2=1
            speed_norm_12=speed_norm_1
        
        U1=self.lam*speed_norm_1+(1-self.lam)*(N1-0.5)/(N1+N2)
        U2=self.lam*speed_norm_2+(1-self.lam)*(N2-0.5)/(N1+N2)

        U12 = self.lam*speed_norm_12+(1-self.lam)*(N1+N2-0.5)/(N1+N2)

        if U12>U1 and U12>U2 and N1+N2<50:
            # Merge groups

        
            newgroup=self.group+other.group
            merged_nogos = self.nogos.union(other.nogos)

            other.is_leader=False
            
            newspeed=min(self.speed,other.speed)
            for ag in newgroup:
                ag.group=newgroup
                ag.nogos=merged_nogos
                ag.speed=newspeed

    def speed_focus(self):
        if self.lam<0.95:
            self.lam+=0.05
        elif self.lam==1:
            pass
        else:
            self.lam=1
        self.is_stuck=False

            
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
    for node, data in G_filtered.nodes(data=True):
        if G.nodes[node].get('type') == 'Camp' and node in paths and G.nodes[node].get('is_open'):
            if data.get('country',None) in ['Niger','Burkina Faso'] and current_date < pd.Timestamp(datetime(2012, 2, 21).date()):
                pass # boarder does not open before 21st feb
            elif node == 'Fassala' and current_date >= pd.Timestamp(datetime(2012, 3, 19).date()):
                pass
            else:
                pops[node]=data.get('population',0)
                paths[node].remove(start_node)
                
                if len(paths[node])>0:
                    trav=G[start_node][paths[node][0]]['travelled']
                else:
                    trav=0
                camp_paths_distances[node] = {'path':paths[node], 'distance':distances[node],'population':pops[node],'travelled':trav}

    
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
                path = nx.shortest_path(G_filtered, start_node, node, weight='weight')
                path.remove(start_node)  # Exclude start node from path
                
                if len(path)>0:
                    trav=G[start_node][path[0]]['travelled']
                else:
                    trav=0
                cities_info[node] = {'distance': path_length, 'path': path, 'population':pops[node],'travelled':trav}
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
    G_filtered = filter_graph_by_max_link_length(G, max_link_length, start_node, nogo=nogo)

    if not G_filtered.has_node(start_node):
        print('Error: Start node is not present in the filtered graph.')
        return {}

    results = {}
    start_country = G.nodes[start_node]['country'] if 'country' in G.nodes[start_node] else 'Not specified'

    border_closure_date = pd.Timestamp(datetime(2012, 2, 21).date())

    for node, data in G_filtered.nodes(data=True):
        node_country = data.get('country')
        node_type = G.nodes[node].get('type')

        if node_country and node_country != start_country and node_type == 'City':
            if node_country in ['Niger', 'Burkina Faso'] and current_date < border_closure_date:
                continue

            try:
                path_length = nx.shortest_path_length(G_filtered, start_node, node, weight='weight')
                path = nx.shortest_path(G_filtered, start_node, node, weight='weight')[1:]  # Skip the start node

                if node not in results or path_length < results[node]['distance']:
                    travelled = G[start_node][path[0]]['travelled'] if path else 0
                    results[node] = {
                        'distance': path_length,
                        'path': path,
                        'population': data.get('population', 0),
                        'travelled': travelled
                    }
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
    
    return death_ids