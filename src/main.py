from tabpfn.model.hyperpc import HyperPC
from tabpfn.data_generation import generate_synth_data
from tabpfn.model.simple_einet.einet import Einet, EinetConfig
from tabpfn.model.simple_einet.layers.distributions.normal import RatNormal

from torch.optim import Adam
import torch
from rtpt import RTPT

def train_hyperpc(epochs: int,
                  lr: float = 0.0001,
                  batch_size: int = 64,
                  min_features: int = 2,
                  max_features: int = 10,
                  num_samples: int = 1000):

    device_id = 0
    device = torch.device(f'cuda:{device_id}')
    hyperpc = HyperPC(device=device_id, nlayers=3).to(torch.device(device))
    optimizer = Adam(hyperpc.parameters(), lr=lr)

    rt = RTPT('JS', 'HyperPC', epochs)
    rt.start()

    x_train, x_test = None, None

    for e in range(epochs):

        print(f"Epoch: {(e + 1)}/{epochs}")

        if e == 0:
            x_train, x_test = generate_synth_data(batch_size, min_feat=min_features, max_feat=max_features, num_samples=num_samples)
        x_train, x_test = x_train.to(device), x_test.to(device)
        
        lls = -torch.sum(hyperpc(train_x=x_train, test_x=x_test))
        print(lls / batch_size)

        optimizer.zero_grad()

        lls.backward()

        optimizer.step()

        rt.step()
    
    return hyperpc

def eval_hyperpc(model: HyperPC, device: int):

    device_ = torch.device(f'cuda:{device}')

    x_train, x_test = generate_synth_data(1, 2, 10, 1000)
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


model = train_hyperpc(20)
eval_hyperpc(model, 0)
