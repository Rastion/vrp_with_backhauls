import math
import random
from qubots.base_problem import BaseProblem
import os

def read_elem(filename):

    # Resolve relative path with respect to this module’s directory.
    if not os.path.isabs(filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, filename)

    with open(filename) as f:
        return f.read().split()

def compute_dist(xi, xj, yi, yj):
    exact_dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
    return int(math.floor(exact_dist + 0.5))

def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    matrix = [[0 for _ in range(nb_customers)] for _ in range(nb_customers)]
    for i in range(nb_customers):
        for j in range(nb_customers):
            if i == j:
                matrix[i][j] = 0
            else:
                d = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
                matrix[i][j] = d
                matrix[j][i] = d
    return matrix

def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    depot_dists = [0 for _ in range(nb_customers)]
    for i in range(nb_customers):
        depot_dists[i] = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
    return depot_dists

def read_input_vrpb(filename):
    # Read tokens from file.
    tokens = read_elem(filename)
    token_iter = iter(tokens)
    
    # Read DIMENSION: total nodes (1 depot + customers)
    while True:
        token = next(token_iter)
        if token.upper() == "DIMENSION":
            next(token_iter)  # Skip the ":"
            nb_nodes = int(next(token_iter))
            nb_customers = nb_nodes - 1
            break

    # Read VEHICLES:
    while True:
        token = next(token_iter)
        if token.upper() == "VEHICLES":
            next(token_iter)  # Skip ":"
            nb_trucks = int(next(token_iter))
            break

    # Read CAPACITY:
    while True:
        token = next(token_iter)
        if token.upper() == "CAPACITY":
            next(token_iter)
            truck_capacity = int(next(token_iter))
            break

    # Read EDGE_WEIGHT_TYPE:
    while True:
        token = next(token_iter)
        if token.upper() == "EDGE_WEIGHT_TYPE":
            next(token_iter)
            edge_weight_type = next(token_iter)
            if edge_weight_type.upper() != "EXACT_2D":
                raise ValueError("Only EXACT_2D edge weight type is supported.")
            break

    # Read NODE_COORD_SECTION:
    while True:
        token = next(token_iter)
        if token.upper() == "NODE_COORD_SECTION":
            break

    depot_x = depot_y = None
    customers_x = []
    customers_y = []
    for n in range(nb_nodes):
        node_id = int(next(token_iter))
        if node_id != n + 1:
            raise ValueError("Unexpected node id in NODE_COORD_SECTION.")
        if node_id == 1:
            depot_x = int(next(token_iter))
            depot_y = int(next(token_iter))
        else:
            x = int(next(token_iter))
            y = int(next(token_iter))
            customers_x.append(x)
            customers_y.append(y)
    dist_matrix_data = compute_distance_matrix(customers_x, customers_y)
    dist_depot_data = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)

    # Read DEMAND_SECTION:
    token = next(token_iter)
    if token.upper() != "DEMAND_SECTION":
        raise ValueError("Expected DEMAND_SECTION.")
    demands = [0 for _ in range(nb_customers)]
    for n in range(nb_nodes):
        node_id = int(next(token_iter))
        if node_id != n + 1:
            raise ValueError("Unexpected node id in DEMAND_SECTION.")
        if node_id == 1:
            if int(next(token_iter)) != 0:
                raise ValueError("Depot demand must be 0.")
        else:
            demands[node_id - 2] = int(next(token_iter))
    
    # Read BACKHAUL_SECTION:
    token = next(token_iter)
    if token.upper() != "BACKHAUL_SECTION":
        raise ValueError("Expected BACKHAUL_SECTION.")
    is_backhaul = {i: False for i in range(nb_customers)}
    while True:
        val = int(next(token_iter))
        if val == -1:
            break
        is_backhaul[val - 2] = True

    # Define delivery and pickup demands.
    delivery_demands = [0 if is_backhaul[i] else demands[i] for i in range(nb_customers)]
    pickup_demands = [demands[i] if is_backhaul[i] else 0 for i in range(nb_customers)]
    
    # Read DEPOT_SECTION:
    token = next(token_iter)
    if token.upper() != "DEPOT_SECTION":
        raise ValueError("Expected DEPOT_SECTION.")
    depot_id = int(next(token_iter))
    if depot_id != 1:
        raise ValueError("Depot id must be 1.")
    end_token = int(next(token_iter))
    if end_token != -1:
        raise ValueError("Expected -1 at end of DEPOT_SECTION.")
    
    return (nb_customers, nb_trucks, truck_capacity, dist_matrix_data, dist_depot_data,
            delivery_demands, pickup_demands, is_backhaul)

class VRPBProblem(BaseProblem):
    """
    Vehicle Routing Problem with Backhauls (VRPB)

    In VRPB, a fleet of vehicles (each with the same capacity) must service customers with two types of demand:
      - Delivery: customers that require goods delivered from the depot.
      - Pickup: customers that send goods back to the depot.
    All customers must be visited exactly once. Moreover, on any vehicle route, all delivery customers
    must be visited before any pickup customer.
    
    The instance is provided in a CVRPLib-like format:
      - DIMENSION: total nodes (1 depot + customers), so number of customers = DIMENSION – 1.
      - VEHICLES: number of trucks available.
      - CAPACITY: vehicle capacity.
      - NODE_COORD_SECTION: node coordinates (first node is the depot).
      - DEMAND_SECTION: demands (depot demand is 0).
      - BACKHAUL_SECTION: lists the pickup nodes (customer ids starting at 2).
      - DEPOT_SECTION: defines the depot.
    
    Decision Variables:
      A candidate solution is represented as a list of routes (one per truck). Each route is an ordered list
      of customer indices (0-indexed, corresponding to the order in which customers were read from the instance).
    
    The model enforces:
      - Each customer is visited exactly once.
      - On any route, if a customer is of pickup type (backhaul), then no preceding delivery (non‑backhaul)
        appears after it.
      - The total delivery and pickup demands on each route do not exceed the truck capacity.
    
    Objective:
      Minimize the total distance traveled by all vehicles, where the distance of a route is computed as:
      depot-to-first + inter-customer distances + last-to-depot.
    """
    def __init__(self, instance_file):
        self.instance_file = instance_file
        (self.nb_customers, self.nb_trucks, self.truck_capacity,
         self.dist_matrix, self.dist_depot, self.delivery_demands,
         self.pickup_demands, self.is_backhaul) = read_input_vrpb(instance_file)
    
    def evaluate_solution(self, candidate) -> float:
        penalty = 0
        total_distance = 0
        
        # Candidate: list of routes (one per truck). Each route is a list of customer indices.
        assigned = []
        for route in candidate:
            assigned.extend(route)
        if sorted(assigned) != list(range(self.nb_customers)):
            penalty += 1e9
        
        for route in candidate:
            if not route:
                continue
            # Check precedence: no delivery (non-backhaul) should precede a pickup.
            for i in range(1, len(route)):
                prev = route[i-1]
                curr = route[i]
                if (not self.is_backhaul[prev]) and self.is_backhaul[curr]:
                    penalty += 1e9
            
            # Capacity constraints:
            route_delivery = sum(self.delivery_demands[j] for j in route)
            route_pickup = sum(self.pickup_demands[j] for j in route)
            if route_delivery > self.truck_capacity:
                penalty += 1e9 * (route_delivery - self.truck_capacity)
            if route_pickup > self.truck_capacity:
                penalty += 1e9 * (route_pickup - self.truck_capacity)
            
            # Compute route distance: from depot to first, between consecutive customers, and from last to depot.
            route_distance = self.dist_depot[route[0]] + self.dist_depot[route[-1]]
            for i in range(1, len(route)):
                route_distance += self.dist_matrix[route[i-1]][route[i]]
            total_distance += route_distance
        
        return total_distance + penalty
    
    def random_solution(self):
        """
        Generates a random candidate solution by randomly partitioning the customers among trucks,
        then randomly shuffling each route.
        """
        customers = list(range(self.nb_customers))
        random.shuffle(customers)
        candidate = [[] for _ in range(self.nb_trucks)]
        for i, cust in enumerate(customers):
            candidate[i % self.nb_trucks].append(cust)
        for route in candidate:
            random.shuffle(route)
        return candidate
