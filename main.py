import os
from dataclasses import dataclass
from typing import Dict, Optional

import click
import flwr
import pandas as pd
import torch
import torchvision
from flwr.client import NumPyClient
from flwr.server import ServerConfig
from torch.utils.tensorboard import SummaryWriter
from hydra.core.hydra_config import HydraConfig
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
from pathlib import Path
from utility import get_parameters, set_parameters

import os
print(os.getcwd())
print(os.listdir())

from datasets import get_dataloaders
from client import test, FlowerNumPyClient
from models import create_model
from scenarios import get_scenario, Scenario
from utility import StaticJudge, StatUtilityJudge

from client_manager import FedZeroCM
from strategy import FedZero

@dataclass
class Experiment:
    scenario: Scenario
    overselect: float
    net_arch: str
    optimizer: str
    opt_args: Dict
    beta: Optional[float]
    proximal_mu: float
    dataset: str

    @property
    def name(self):
        aggregation_strategy = "FedAvg"
        iid_str = "noniid" if self.beta is None else f"b={self.beta:.1f}"
        scenario_str = "no_constr" if self.scenario.unconstrained else self.scenario.solar_scenario
        imbalanced_str = "_imbalanced" if self.scenario.imbalanced_scenario else ""
        overselect_str = f"_{self.overselect:.1f}K" if self.overselect > 1 else ""
        error_str = ""
        # if "fedzero" in str(self.selection_strategy) and self.scenario.forecast_error != "error":
        #     error_str = f",{self.scenario.forecast_error}"

        experiment_name = (f"{scenario_str}{imbalanced_str},"
                           f"{self.dataset},{iid_str},{self.net_arch},"
                           f"{aggregation_strategy},"
                           f"{overselect_str}{error_str}")

        i = 0
        while os.path.exists(f"runs/{experiment_name},i={i}"):
            i += 1
        return experiment_name + f",i={i}"


def get_model_and_hyperparameters(dataset, iid):
    optimizer = "SGD"
    if dataset == "cifar10":
        net_arch = 'resnet18'
        net_arch_size_factor = 1
        opt_args = {'lr': 0.001, 'weight_decay': 5e-4, 'momentum': 0.9}
        if iid:
            proximal_mu = 0
            beta = 1
        else:
            proximal_mu = 0.1
            beta = 0.5
    elif dataset == "mnist":
        net_arch = 'mlp'
        net_arch_size_factor = 1
        opt_args = {'lr': 0.01, 'weight_decay': 0, 'momentum': 0}
        if iid:
            proximal_mu = 0
            beta = 1
        else:
            proximal_mu = 0.1
            beta = 0.5
    elif dataset == "femnist":
        net_arch = 'cnn'
        net_arch_size_factor = 1
        opt_args = {'lr': 0.01, 'weight_decay': 0, 'momentum': 0}
        if iid:
            proximal_mu = 0
            beta = 1
        else:
            proximal_mu = 0.1
            beta = 0.5
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return net_arch, net_arch_size_factor, optimizer, opt_args, proximal_mu, beta


def simulate_fl_training(experiment: Experiment, device: torch.device, cfg: DictConfig) -> None:
    print(f"Starting experiment {experiment.name} ...")
    writer = SummaryWriter(log_dir="runs/"+experiment.name)

    os.makedirs(f'trained_models/{experiment.name}/', exist_ok=True)

    trainloaders, testloader, num_classes = get_dataloaders(
        dataset=experiment.dataset,
        num_clients=cfg.Simulation['NUM_CLIENTS'],
        batch_size=cfg.Simulation['BATCH_SIZE'],
        beta=experiment.beta,
        cfg=cfg
    )

    print(f"Sample distribution: {pd.Series([len(t.batch_sampler.sampler) for t in trainloaders]).describe()}")

    # Initialize 1 model for initial params
    model = create_model(cfg=cfg.Scenario, model_rate=1, device=device)
    # initial_params = get_parameters(model)

    for i, (c, trainloader) in enumerate(zip(experiment.scenario.client_load_api.get_clients(), trainloaders)):
        c.num_samples = len(trainloader) * cfg.Simulation['BATCH_SIZE']
        required_time = c.num_samples / (c.batches_per_timestep * cfg.Simulation['TIMESTEP_IN_MIN'])
        # if required_time <= 5 or required_time >= 55:
        print(f"{i+1:>3}: {required_time:.0f} mins ({len(trainloader)} batches at {c.batches_per_timestep:.1f} batches/min)")

    def client_fn(client_name) -> NumPyClient:
        client_id = int(client_name.split('_')[0])
        return FlowerNumPyClient(client_name=client_name,
                                train_loader=trainloaders[client_id],
                                cfg=cfg,
                                device=device)

    # The `evaluate` function will be by Flower called after every round
    def server_eval_fn(server_round: int, parameters: flwr.common.NDArrays, config: Dict[str, flwr.common.Scalar]):
        net = create_model(cfg=cfg.Scenario, model_rate=1, device=device)
        set_parameters(net, parameters)  # Update model with the latest parameters
        loss, accuracy = test(net, testloader, device=device)
        net_state_dict = net.state_dict()
        if cfg.Simulation['SAVE_TRAINED_MODELS'] and net_state_dict is not None:
            torch.save(net_state_dict, f"trained_models/{experiment.name}/round_{server_round}")
        print(f"Server-side evaluation, round: {server_round},  loss: {loss},  accuracy: {accuracy}")
        return loss, {"accuracy": accuracy}

   
    model_rates = [1, 0.5, 0.25, 0.125, 0.0625]
    client_to_param_index = {i: [v.shape for _, v in create_model(cfg.Scenario, i).state_dict().items()] for i in model_rates}
    client_to_batches = [len(client_train_loader) for client_train_loader in trainloaders]

    client_manager = FedZeroCM(experiment.scenario.power_domain_api, experiment.scenario.client_load_api, experiment.scenario, cfg, client_to_batches)
    
    pretrained_model = torchvision.models.resnet18(weights='DEFAULT')
    pretrained_model.fc = torch.nn.Linear(pretrained_model.fc.in_features, 10)
    custom_model = create_model(cfg.Scenario, model_rate = 1)

    # Load compatible layers
    custom_model.load_state_dict({k: v for k, v in zip(custom_model.state_dict().keys(), pretrained_model.state_dict().values())}, strict=True)
    initial_params = get_parameters(custom_model)

    strategy = FedZero(
        client_to_param_index=client_to_param_index,
        model = custom_model,
        fraction_fit=cfg.Simulation['NUM_CLIENTS'] / cfg.Simulation['CLIENTS_PER_ROUND'],
        fraction_evaluate=0,  # we only do server side evaluation
        initial_parameters=flwr.common.ndarrays_to_parameters(initial_params),
        evaluate_fn=server_eval_fn
    )

    history = flwr.simulation.start_simulation(
        client_fn=client_fn,
        clients_ids=[c.name for c in experiment.scenario.client_load_api.get_clients()],
        client_manager=client_manager,
        strategy=strategy,
        config=ServerConfig(num_rounds=cfg.Simulation['MAX_ROUNDS']),
        client_resources= {
            'num_cpus' : cfg.RAY_CLIENT_RESOURCES['num_cpus'],
            'num_gpus' : cfg.RAY_CLIENT_RESOURCES['num_gpus']
        }
    )
    print("Simulation finished successfully.")

    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    results = {
        "history": history,
        "config": cfg
    }

    with open(results_path, "wb") as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)



@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    assert cfg.Scenario['overselect'] >= 1
    clients_per_round = int(cfg.Simulation['CLIENTS_PER_ROUND'] * cfg.Scenario['overselect'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")

    net_arch, net_arch_size_factor, optimizer, opt_args, proximal_mu, beta = get_model_and_hyperparameters(cfg.Scenario['dataset'], iid=False)

    if "fedzero" in cfg.Scenario['approach']:
        split = cfg.Scenario['approach'].split("_")
        assert len(split) == 3, ("Invalid approach format: FedZero has the format fedzero_{alpha}_{exclusion_factor}, "
                                 "e.g. fedzero_1_1")
        # selection_strategy = FedZeroSelectionStrategy(
        #     clients_per_round=clients_per_round,
        #     utility_judge=StatUtilityJudge(scenario.client_load_api.get_clients()),
        #     alpha=float(split[1]),
        #     exclusion_factor=float(split[2]),
        #     min_epochs=MIN_LOCAL_EPOCHS,
        #     max_epochs=MAX_LOCAL_EPOCHS,
        #     seed=seed,
        # )
    else:
        raise click.ClickException(f"Unknown approach: {cfg.approach}")
    
    scenario = get_scenario(cfg.Scenario['scenario'],
                            net_arch_size_factor=net_arch_size_factor,
                            forecast_error=cfg.Scenario['forecast_error'],
                            imbalanced_scenario=cfg.Scenario['imbalanced_scenario'],
                            cfg = cfg
                            )

    experiment = Experiment(scenario=scenario,
                            overselect=cfg.Scenario['overselect'],
                            net_arch=net_arch,
                            optimizer=optimizer,
                            opt_args=opt_args,
                            beta=beta,
                            proximal_mu=proximal_mu,
                            dataset=cfg.Scenario['dataset'])
    simulate_fl_training(experiment, device, cfg)


if __name__ == "__main__":
    main()