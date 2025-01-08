from .utils import get_batch_to_dataloader
from tabpfn.utils import default_device
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import random

# Step 1: Generate a random DAG
def generate_random_dag(num_nodes, edge_prob=0.2):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                dag.add_edge(i, j)
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Generated graph is not a DAG.")
    return dag

# Step 2: Initialize random vectors for root nodes
def initialize_root_values(dag, vector_dim):
    values = {}
    for node in dag.nodes:
        if not list(dag.predecessors(node)):  # Root node has no parents
            values[node] = torch.randn(vector_dim)
    return values

# Step 3: Compute values for non-root nodes using random MLPs
class RandomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

def compute_node_values(dag, values, max_hidden_dim, min_hidden_dim):
    hidden_dim = np.random.choice(list(range(min_hidden_dim, max_hidden_dim)))
    
    for node in nx.topological_sort(dag):  # Process nodes in topological order
        if node in values:
            continue
        input_dim = len(nx.predecessor(dag, node))
        mlp = RandomMLP(input_dim=input_dim, hidden_dim=hidden_dim)
        parent_values = torch.cat([values[parent].unsqueeze(0) for parent in dag.predecessors(node)], dim=-1)
        values[node] = mlp(parent_values)
    return values

def get_batch():
    num_nodes = np.random.randint(5, 15)
    batch_size = 64  # Dimensionality of the vectors
    dag = generate_random_dag(num_nodes, edge_prob=0.3)

    # Step 2: Initialize root node values
    values = initialize_root_values(dag, batch_size)

    # Step 3: Compute values for remaining nodes
    values = compute_node_values(dag, values, min_hidden_dim=2, max_hidden_dim=10)

    return values


DataLoader = get_batch_to_dataloader(get_batch)
