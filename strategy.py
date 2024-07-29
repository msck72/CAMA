"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from models import copy_gp_to_lp
import numpy as np
import copy
import torch
from collections import OrderedDict

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy

from models import get_state_dict_from_param


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# pylint: disable=line-too-long
class FedZero(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        client_to_param_index,
        model,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.client_to_param_index = client_to_param_index
        self.model = model

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace

        self.prev_utility = []

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        # fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients, server_round=server_round, prev_rnd_clnts_stat_util = self.prev_utility
        )

        global_param_with_sd = get_state_dict_from_param(self.model, parameters)

        final_list_of_clients = []
        for client in clients:
            print(client.properties)
            client_parameters = copy_gp_to_lp(global_param_with_sd, self.client_to_param_index[client.properties['model_rate']])
            # fit_ins = adapted_model_parameters(client, parameters)
            local_param_fitres = [v.cpu() for v in client_parameters.values()]
            final_list_of_clients.append((client, FitIns(ndarrays_to_parameters(local_param_fitres), client.properties)))
        # Return client/config pairs
        # return [(client, fit_ins) for client in clients]
        print("completed configure fit")
        return final_list_of_clients

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients,
        )
        # here we need to define the classes it belongs to and its properties accordingly
        # write a loop (for every client selected get its capacity and return the adapted model)
        # Return client/config pairs

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        global_parameters = self.model.state_dict()
        label_split_all = [fit_res.metrics['label_split'].type(torch.int) for _, fit_res in results]
        num_examples = [fit_res.num_examples for _, fit_res in results]

        local_parameters = _create_local_parameters(self.model.state_dict().keys(), results)
        
        param_idx = _calculate_param_idx(self.model.state_dict(), local_parameters)

        count = OrderedDict()
        for k, v in global_parameters.items():
            #   print(f'{k}   {v.shape}')
              parameter_type = k.split('.')[-1]
              count[k] = v.new_zeros(v.size(), dtype=torch.float32)
              tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
              for m in range(len(local_parameters)):
                  if 'weight' in parameter_type or 'bias' in parameter_type:
                      if parameter_type == 'weight':
                          if v.dim() > 1:
                              if 'linear' in k:
                                  label_split = label_split_all[m]
                                  param_idx[m][k] = list(param_idx[m][k])
                                  # print(type(param_idx[m][k][0]))
                                #   print('label_split = ', label_split)
                                #   print(f'{type(param_idx[m][k][0])}, {param_idx[m][k][0]}')
                                  param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                #   print(f'{type(label_split)}, {label_split}')
                                #   print(param_idx[m][k])
                                  tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                  count[k][torch.meshgrid(param_idx[m][k])] += num_examples[m]
                              else:
                                  tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                  count[k][torch.meshgrid(param_idx[m][k])] += num_examples[m]
                          else:
                              tmp_v[param_idx[m][k]] += local_parameters[m][k]
                              count[k][param_idx[m][k]] += num_examples[m]
                      else:
                          if 'linear' in k:
                              label_split = label_split_all[m]
                            #   print(f'{type(label_split)}, {label_split}')
                            #   print(param_idx[m][k])
                              param_idx[m][k] = param_idx[m][k][0]
                              param_idx[m][k] = param_idx[m][k][label_split]
                              tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                              count[k][param_idx[m][k]] += num_examples[m]
                          else:
                              tmp_v[param_idx[m][k]] += local_parameters[m][k]
                              count[k][param_idx[m][k]] += num_examples[m]
                  else:
                      tmp_v[param_idx[m][k]] += local_parameters[m][k]
                      count[k][param_idx[m][k]] += num_examples[m]
              tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
              v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)

        self.prev_utility = [(cp.cid, fit_res.metrics['statistical_utility']) for cp, fit_res in results]

        global_values = []
        for v in self.model.state_dict().values():
            global_values.append(v.numpy())

        return ndarrays_to_parameters(global_values), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
    

def _calculate_param_idx(global_model, local_models):
    param_idx = {}
    for m, local_model in enumerate(local_models):
        param_idx[m] = {}
        for (name, global_param), (_, local_param) in zip(global_model.items(), local_model.items()):
            if global_param.shape == local_param.shape:
                param_idx[m][name] = tuple(torch.arange(s) for s in global_param.shape)
            else:
                indices = []
                for g_dim, l_dim in zip(global_param.shape, local_param.shape):
                    if g_dim == l_dim:
                        indices.append(torch.arange(g_dim))
                    else:
                        # start = (g_dim - l_dim) // 2
                        start = l_dim
                        indices.append(torch.arange(0, l_dim))
                param_idx[m][name] = tuple(indices)
    return param_idx




def _create_local_parameters(global_model_keys, results):
    local_parameters = [OrderedDict() for _ in results]

    for i, (_, fit_res) in enumerate(results):
        for k , v in zip(global_model_keys, parameters_to_ndarrays(fit_res.parameters)):
            local_parameters[i][k] = v * fit_res.num_examples

    return local_parameters