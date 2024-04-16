import matplotlib.pyplot as plt
import networkx as nx

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
                   population=loc.population)
        
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
