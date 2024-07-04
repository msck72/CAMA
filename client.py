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
    """Standard Flower client for training."""

    def __init__(
        self,
        model_rate: Optional[float] = 1,
        train_loader = None,
        test_loader = None,
        cfg = None,
        device = torch.device("cpu")
    ):
        self.model_rate = model_rate,
        self.train_loader = train_loader,
        self.test_loader = test_loader,
        self.cfg = cfg,
        self.model = None
        self.device = device

    def get_parameters(self, config) -> NDArrays:
        """Return the parameters of the current net."""
        return get_parameters(self.model) if self.model is not None else None

    def fit(self, parameters, config) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        # print(f"cid = {self.cid}")
        # create the model here... with the model rate in the config...
        self.model = create_model(self.cfg)
        set_parameters(self.model, parameters)
        train(self.model, self.train_loader, self.label_split, self.cfg)
        return get_parameters(self.model), len(self.trainloader), {'model_rate': config['model_rate']}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        # create the model here... with the model rate in the config...
        # self.model = create_model(model_rate = config['model_rate], )
        self.model = create_model(self.cfg)
        set_parameters(self.model, parameters)
        loss, accuracy = test(
            self.model, self.test_loader, device=self.device
        )
        return float(loss), len(self.test_loader), {"accuracy": float(accuracy)}



def train(model, train_loader, label_split, settings, device):
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = make_optimizer()

    model.train()
    for _ in range(settings.epochs):
        for _, input in enumerate(train_loader):
            input_dict = {}
            input_dict["img"] = input[0].to(device)
            input_dict["label"] = input[1].to(device)
            input_dict["label_split"] = label_split.type(torch.int).to(device)
            optimizer.zero_grad()
            output = model(input_dict)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
    return



def test(model, test_loader, label_split = None, device=torch.device("cpu")):
    model.eval()
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(test_loader):
            input_dict = {}
            input_dict["img"] = input[0].to(device)
            input_dict["label"] = input[1].to(device)
            if label_split != None:
                input_dict["label_split"] = label_split
            output = model(input_dict)
            test_loss += output["loss"].item()
            correct += (
                (output["score"].argmax(1) == input_dict["label"])
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss /= num_batches
    correct /= size
    return test_loss, correct