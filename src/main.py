from tabpfn.model.hyperpc import HyperPC
from tabpfn.data_generation import generate_synth_data, generate_synth_data_interventions
from tabpfn.model.simple_einet.einet import Einet, EinetConfig
from tabpfn.model.simple_einet.layers.distributions.normal import RatNormal

from torch.optim import Adam
import torch
from rtpt import RTPT
import matplotlib.pyplot as plt


def train_hyperpc(epochs: int,
                  lr: float = 0.0001,
                  batch_size: int = 64,
                  min_features: int = 2,
                  max_features: int = 10,
                  num_samples: int = 1000,
                  device_id: int = 0):

    device = torch.device(f'cuda:{device_id}')
    hyperpc = HyperPC(device=device_id, nlayers=3).to(torch.device(device))
    optimizer = Adam(hyperpc.parameters(), lr=lr)

    rt = RTPT('FB', 'HyperPC', epochs)
    rt.start()

    x_train, x_test = None, None

    for e in range(epochs):

        print(f"Epoch: {(e + 1)}/{epochs}")

        if e == 0:
            x_train, x_test = generate_synth_data_interventions(batch_size, min_feat=min_features, max_feat=max_features, num_samples=num_samples)
        x_train, x_test = x_train.to(device), x_test.to(device)
        
        lls = -torch.sum(hyperpc(train_x=x_train, test_x=x_test))
        print(lls / batch_size)

        optimizer.zero_grad()

        lls.backward()

        optimizer.step()

        rt.step()
    
    return hyperpc


def visual_eval(model: HyperPC, x_train, x_test, device):
    indices = torch.nonzero(x_train[0, 0, :] != 0).squeeze()
    if indices.numel() == 0:
        print("No Intervention")
        intervention = None
    else:
        assert len(indices.shape) == 0
        intervention = indices.item()
        print("Intervention: ", intervention)
    has_data = [torch.all(x_train[:, :, i] == 0).item() for i in range(x_train.shape[2])]
    no_data_indices = [i for i, hd in enumerate(has_data) if not hd]
    print("Data indices: ", no_data_indices)

    # model samples
    params = model(train_x=x_train, test_x=x_test, return_params = True)
    assert x_test.shape[1] == 1 and len(params) == 1
    params = params[0]
    steps = 1000
    model_predictions = {}
    for i in range(x_test.shape[2]):
        min_value = torch.min(x_test[:, 0, i]).item()
        max_value = torch.max(x_test[:, 0, i]).item()
        x = torch.linspace(min_value, max_value, steps).to(device)
        x_input = torch.empty(steps, x_train.shape[1], x_train.shape[2]).to(device)
        x_input[:, 0, i] = x
        marginalized_scopes = torch.arange(0, x_train.shape[2]).to(device)
        marginalized_scopes = marginalized_scopes[marginalized_scopes != i]
        y = model.einet(x_input, params, marginalized_scopes=marginalized_scopes)
        model_predictions[i] = (x, torch.exp(y))


    x_train = x_train.cpu().detach().numpy()
    x_test = x_test.cpu().detach().numpy()

    # ground truth input and interventional distribution
    # loop over features
    for i in range(x_train.shape[2]):
        obs_data = x_train[:, 0, i]
        int_data = x_test[:, 0, i]

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].hist(obs_data, bins=50, alpha=0.5, color='blue', label='Observed Data', density=True)
        axs[0].legend()

        axs[1].hist(int_data, bins=50, alpha=0.5, color='red', label='Intervention Data', density=True)
        axs[1].legend()
        model_x = model_predictions[i][0].cpu().detach().numpy()
        model_y = model_predictions[i][1].cpu().detach().numpy()
        axs[1].plot(model_x, model_y, color='black', label='Model')

        # Ensure the same x-axis is used for both plots
        min_x = min(axs[0].get_xlim()[0], axs[1].get_xlim()[0])
        max_x = max(axs[0].get_xlim()[1], axs[1].get_xlim()[1])
        axs[0].set_xlim(min_x, max_x)
        axs[1].set_xlim(min_x, max_x)
        # Ensure the same y-axis is used for both plots
        axs[1].set_ylim(axs[0].get_ylim())

        plt.savefig(f'evaluation_plot_{i}.pdf')


def eval_hyperpc(model: HyperPC, device: int):

    device_ = torch.device(f'cuda:{device}')

    x_train, x_test = generate_synth_data_interventions(1, 2, 10, 10000, train_test_split=0.5)
    x_train, x_test = x_train.to(device), x_test.to(device_)

    lls = torch.sum(model(train_x=x_train, test_x=x_test))

    #einet_cfg = EinetConfig(10, depth=1, num_channels=1, leaf_type=RatNormal)
    #einet = Einet(einet_cfg).to(device_)

    #optimizer = Adam(einet.parameters(), 0.01)

    #for e in range(500):
    #    optimizer.zero_grad()

    #    einet_lls = -torch.sum(einet(x_train))

    #    einet_lls.backward()

    #    optimizer.step()

    #einet_lls = torch.sum(einet(x_test))

    print(f"HyperPC LLs: {lls}")
    #print(f"Einet LLS: {einet_lls}")

    # do plots showing the marginal distribution of obs and int data
    visual_eval(model, x_train, x_test, device_)

device_id = 1
model = train_hyperpc(100, device_id=device_id)
eval_hyperpc(model, device_id)

# TODOs
# implement sampling
# different noise distributions
# make it so that graphs do not always follow the causal order given by the index order