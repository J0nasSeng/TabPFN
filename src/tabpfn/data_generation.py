import torch
import networkx as nx
import numpy as np
import torch.nn as nn
import random

# Step 1: Create a random DAG
def generate_random_dag(num_nodes, edge_prob=0.3):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Ensure DAG property (edges go forward)
            if random.random() < edge_prob:
                dag.add_edge(i, j)

    return dag

# Step 2: Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        hidden_size = max(input_size // 2, 4)
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.model(x)

# Step 3: Process the DAG and generate data
def generate_dag_data(dag, num_samples=1000):
    node_values = {}  # Stores computed values for each node
    neural_nets = {}  # Stores neural networks for nodes with parents

    for node in nx.topological_sort(dag):  # Ensure parents are processed first
        parents = list(dag.predecessors(node))
        
        if parents:  # If the node has parents, use a neural network
            input_data = torch.cat([node_values[p] for p in parents], dim=-1)
            if node not in neural_nets:
                neural_nets[node] = SimpleNN(input_size=len(parents))
            node_values[node] = neural_nets[node](input_data)
        else:  # If the node has no parents, draw from a Gaussian
            node_values[node] = torch.randn(num_samples).unsqueeze(1)

    # Combine all node values into a tensor
    all_values = torch.stack([node_values[node] for node in sorted(node_values.keys())]).squeeze().T
    return all_values


def generate_synth_data(batch_size: int, 
                        min_feat: int = 2, 
                        max_feat:int = 10, 
                        num_samples: int = 1000,
                        train_test_split: float = 0.8):

    datasets = []

    for b in range(batch_size):
        feats = np.random.randint(min_feat, max_feat)
        dag = generate_random_dag(feats)
        data = generate_dag_data(dag, num_samples)

        # pad features
        if data.shape[1] < max_feat:
            data = torch.nn.functional.pad(data, (0, max_feat - data.shape[1]), mode='constant', value=0)
        
        datasets.append(data)

    datasets = torch.stack(datasets)
    train_samples = int(datasets.shape[1] * train_test_split)
    x_train, x_test = datasets[:, :train_samples, :], datasets[:, train_samples:, :]
    return x_train.permute(1, 0, 2), x_test.permute(1, 0, 2)
