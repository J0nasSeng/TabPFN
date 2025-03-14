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
        o = self.model(x)
        return self.add_noise(o)
    
    def add_noise(self, o):
        # add gaussian noise
        return o + torch.randn_like(o) * 0.1

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

def generate_dag_data_interventions(dag, num_samples_train=800, num_samples_test=200):
    node_values = {}  # Stores computed values for each node
    node_values_post_intervention = {}  # Stores computed values for each node post intervention
    neural_nets = {}  # Stores neural networks for nodes with parents

    nr_nodes = len(dag.nodes)
    if random.random() < 0.5:
        # observational data
        intervention_index = -1
    else:
        # interventional data
        intervention_index = torch.randint(0, nr_nodes, (1,)).item()
    intervention_value = torch.randn(1).item()  # TODO different intervention values? bigger range?

    for node in nx.topological_sort(dag):  # Ensure parents are processed first
        parents = list(dag.predecessors(node))
        
        if parents:  # If the node has parents, use a neural network
            input_data = torch.cat([node_values[p] for p in parents], dim=-1)
            if node not in neural_nets:
                neural_nets[node] = SimpleNN(input_size=len(parents))
            samples = neural_nets[node](input_data)
            node_values[node] = samples
            # interventional data
            input_data = torch.cat([node_values_post_intervention[p] for p in parents], dim=-1)
            samples = neural_nets[node](input_data)
            if node != intervention_index:
                node_values_post_intervention[node] = samples
            else:
                node_values_post_intervention[node] = torch.full_like(samples, intervention_value)
        else:  # If the node has no parents, draw from a Gaussian
            samples = torch.randn(num_samples_train).unsqueeze(1)
            node_values[node] = samples
            # interventional data
            samples = torch.randn(num_samples_test).unsqueeze(1)  # TODO use different data? Could also use the same, then you also have counterfactuals... I don't think it matters for interventions
            if node != intervention_index:
                node_values_post_intervention[node] = samples
            else:
                node_values_post_intervention[node] = torch.full_like(samples, intervention_value)

    # Combine all node values into a tensor
    all_values = torch.stack([node_values[node] for node in sorted(node_values.keys())]).squeeze().T
    all_values_post_intervention = torch.stack([node_values_post_intervention[node] for node in sorted(node_values_post_intervention.keys())]).squeeze().T
    all_values = torch.cat([torch.zeros(1, all_values.shape[1]), all_values], dim=0)
    if intervention_index != -1:
        all_values[0, intervention_index] = intervention_value
    return all_values, all_values_post_intervention

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


def generate_synth_data_interventions(batch_size: int, 
                        min_feat: int = 2, 
                        max_feat:int = 10, 
                        num_samples: int = 1000,
                        train_test_split: float = 0.8):

    datasets = []
    datasets_int = []
    num_samples_train = int(num_samples * train_test_split)
    num_samples_test = num_samples - num_samples_train

    for b in range(batch_size):
        feats = np.random.randint(min_feat, max_feat)
        dag = generate_random_dag(feats)
        if batch_size == 1:
            print(dag.edges)
        data, data_int = generate_dag_data_interventions(dag, num_samples_train, num_samples_test)

        # pad features
        if data.shape[1] < max_feat:
            data = torch.nn.functional.pad(data, (0, max_feat - data.shape[1]), mode='constant', value=0)
        if data_int.shape[1] < max_feat:
            data_int = torch.nn.functional.pad(data_int, (0, max_feat - data_int.shape[1]), mode='constant', value=0)
        
        datasets.append(data)
        datasets_int.append(data_int)

    x_train = torch.stack(datasets)
    x_test = torch.stack(datasets_int)
    return x_train.permute(1, 0, 2), x_test.permute(1, 0, 2)
