"""Defines the MNIST Flower Client and a function to instantiate it."""

from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays
import numpy as np
import torch.nn.functional as F
from utility import get_parameters, set_parameters, make_optimizer
from models import create_model


class FlowerNumPyClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_name,
        train_loader,
        test_loader = None,
        cfg = None,
        device = torch.device("cpu")
    ):
        print(f'I am client {client_name}')
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.model = None
        self.device = device

        all_labels = []
        for batch in self.train_loader:
            # Assuming batch is a tuple (input, labels)
            # If it's a different structure, adjust accordingly
            labels = batch[1]
            all_labels.extend(labels.numpy())

        self.label_split = torch.Tensor(list(set(all_labels)))

    def get_parameters(self, config) -> NDArrays:
        """Return the parameters of the current net."""
        return get_parameters(self.model) if self.model is not None else None

    def fit(self, parameters, config) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        # print(f"cid = {self.cid}")
        # create the model here... with the model rate in the config...
        self.model = create_model(self.cfg.Scenario, model_rate=config['model_rate'], device=self.device, track=self.cfg.Scenario.track)
        set_parameters(self.model, parameters)
        stat_util = train(self.model, self.train_loader, self.label_split, self.cfg, self.device )
        return get_parameters(self.model), len(self.train_loader), {'model_rate': config['model_rate'], 'label_split': self.label_split, 'statistical_utility': stat_util}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        # create the model here... with the model rate in the config...
        # self.model = create_model(model_rate = config['model_rate], )
        self.model = create_model(self.cfg.Scenario, model_rate=config['model_rate'])
        set_parameters(self.model, parameters)
        loss, accuracy = test(
            self.model, self.test_loader, device=self.device
        )
        return float(loss), len(self.test_loader), {"accuracy": float(accuracy)}



# def train(model, train_loader, label_split, settings, device):
#     # criterion = torch.nn.CrossEntropyLoss()
#     optimizer = make_optimizer(settings, model.parameters())

#     model.train()
#     for _ in range(settings.Simulation['EPOCHS']):
#         for _, input in enumerate(train_loader):
#             input_dict = {}
#             input_dict["img"] = input[0].to(device)
#             input_dict["label"] = input[1].to(device)
#             input_dict["label_split"] = label_split.type(torch.int).to(device)
#             optimizer.zero_grad()
#             output = model(input_dict)
#             output["loss"].backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
#             optimizer.step()
#     return


def train(model, train_loader, label_split, settings, device):

    optimizer = make_optimizer(settings, model.parameters())

    model.train()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    sample_loss = []
    local_round_loss, local_round_acc, total = 0.0, 0.0, 0
    input_dict = {}
    input_dict["label_split"] = label_split.type(torch.int).to(device)
    
    for _ in range(settings.Simulation['EPOCHS']):
        for _, data in enumerate(train_loader):
            X, Y = data[0].to(device), data[1].to(device)
            input_dict['img'] = X
            optimizer.zero_grad()
            outputs = model(input_dict)
            individual_sample_loss = criterion(outputs, Y)  # required for Oort statistical utility
            sample_loss.extend(individual_sample_loss)
            ce_loss = torch.mean(individual_sample_loss)

            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            local_round_acc += (predicted == Y).sum().item()

            loss = ce_loss
            loss.backward()
            local_round_loss += loss.item()
            optimizer.step()
    
    local_round_loss /= total
    local_round_acc /= total

    return statistical_utility(sample_loss)



def statistical_utility(sample_loss: list):
    """Statistical utility as defined in Oort"""
    squared_sum = sum([torch.square(l) for l in sample_loss]).item()
    return len(sample_loss) * np.sqrt(1/len(sample_loss) * squared_sum)


# def test(model, test_loader, label_split = None, device=torch.device("cpu")):
#     model.eval()
#     size = len(test_loader.dataset)
#     num_batches = len(test_loader)
#     test_loss, correct = 0, 0

#     with torch.no_grad():
#         model.train(False)
#         for i, input in enumerate(test_loader):
#             input_dict = {}
#             input_dict["img"] = input[0].to(device)
#             input_dict["label"] = input[1].to(device)
#             if label_split != None:
#                 input_dict["label_split"] = label_split
#             output = model(input_dict)
#             test_loss += output["loss"].item()
#             correct += (
#                 (output["score"].argmax(1) == input_dict["label"])
#                 .type(torch.float)
#                 .sum()
#                 .item()
#             )

#     test_loss /= num_batches
#     correct /= size
#     return test_loss, correct


def test(model, test_loader, label_split = None, device=torch.device("cpu")):
    """Validate the network on the entire test set."""
    model.eval()
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loss, accuracy, total = 0, 0, 0.0
    
    input_dict = {}
    if label_split != None:
        input_dict["label_split"] = label_split

    with torch.no_grad():
        for data in test_loader:
            input_dict['img'] = data[0].to(device)
            Y = data[1].to(device)
            outputs = model(input_dict)
            loss += criterion(outputs, Y).item()
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            accuracy += (predicted == Y).sum().item()
    loss /= total
    accuracy /= total
    return loss, accuracy