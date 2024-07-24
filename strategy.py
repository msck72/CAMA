"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from models import copy_gp_to_lp
import numpy as np
import copy
import torch

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
        print(f'In Aggregate Tiger Nageswara rao and number of client proxys = {len(results)}')

        def aggregate_layer(layer_updates):
            # print("in aggregate in-between layers")
            """Padding layers with 0 to max size, then average them"""
            # Get the layer's largest form
            if all(l.shape == () for l in layer_updates):
                return np.array(np.mean(layer_updates))
            
            max_ch = np.max([np.shape(l) for l in layer_updates], axis=0)
            layer_agg = np.zeros(max_ch)
            count_layer = np.zeros(max_ch)  # to average by num of models that size
            for l in layer_updates:
                local_ch = np.shape(l)
                pad_shape = [(0, a) for a in (max_ch - local_ch)]
                # print(f'l_shape = {l.shape}')
                # print(f'pad_shape = {pad_shape}')
                l_padded = np.pad(l, pad_shape, constant_values=0.0)
                ones_of_shape = np.ones(local_ch)
                ones_pad = np.pad(ones_of_shape, pad_shape, constant_values=0.0)
                count_layer = np.add(count_layer, ones_pad)
                layer_agg = np.add(layer_agg, l_padded)
            if np.any(count_layer == 0.0):
                print(count_layer)
                raise ValueError("Diving with 0")
            layer_agg = layer_agg / count_layer
            return layer_agg


        def aggregate_layer_weight(layer_updates):
            # print("in aggregate in-between layers")
            """Padding layers with 0 to max size, then average them"""
            # Get the layer's largest form
            max_ch = np.max([np.shape(l) for l in layer_updates], axis=0)
            layer_agg = np.zeros(max_ch)
            count_layer = np.zeros(max_ch)  # to average by num of models that size
            for m , l in enumerate(layer_updates):              
                local_ch = np.shape(l)
                pad_shape = [(0, a) for a in (max_ch - local_ch)]
                l_padded = np.pad(l, pad_shape, constant_values=0.0)
                
                ones_of_shape = np.ones(local_ch) 
                ones_pad = np.pad(ones_of_shape, pad_shape, constant_values=0.0)
                count_layer = np.add(count_layer, ones_pad)
                layer_agg = np.add(layer_agg, l_padded)
            count_layer[count_layer == 0] = 1
            if np.any(count_layer == 0.0):
                print(count_layer)
                raise ValueError("Diving with 0")
            layer_agg = layer_agg / count_layer
            # print(f'aggregated layer = {layer_agg}')
            return layer_agg

        
        def aggregate_last_layer(layer_updates, label_split):
            # print('in aggregate last layer')
            """Padding layers with 0 to max size, then average them"""
            # Get the layer's largest form
            max_ch = np.max([np.shape(l) for l in layer_updates], axis=0)
            layer_agg = np.zeros(max_ch)
            count_layer = np.zeros(max_ch)  # to average by num of models that size
            for m , l in enumerate(layer_updates):
                local_ch = np.shape(l)
                label_mask = np.zeros(local_ch)
                label_mask[label_split[m].type(torch.int).numpy()] = 1
                l = l * label_mask
                ones_of_shape = np.ones(local_ch)
                ones_of_shape =  ones_of_shape * label_mask
                count_layer = np.add(count_layer, ones_of_shape)
                layer_agg = np.add(layer_agg, l)
            count_layer[count_layer == 0] = 1
            if np.any(count_layer == 0.0):
                raise ValueError("Diving with 0")
            layer_agg = layer_agg / count_layer
            return layer_agg


        self.prev_utility = [(cp.cid, fit_res.metrics['statistical_utility']) for cp, fit_res in results]
        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer for layer in parameters_to_ndarrays(fit_res.parameters)] for _ , fit_res in results
        ]

        global_values = []
        for v in self.model.state_dict().values():
            global_values.append(copy.deepcopy(v))

        num_layers = len(weighted_weights[0])

        label_split = [fit_res.metrics['label_split'].type(torch.int) for _, fit_res in results]

        flat_label_split = torch.cat([t.view(-1) for t in label_split])
        unique_labels = torch.unique(flat_label_split)
        all_labels = torch.arange(10)
        missing_labels = torch.tensor(list(set(all_labels.tolist()) - set(unique_labels.tolist())))
        print(f'missing labels = {missing_labels}')


        agg_layers = []
        last_weight_layer = [None for i in range(10)]
        for i, l in enumerate(zip(*weighted_weights), start=1):
            if( i == num_layers - 1):
                
                for i in range(10):
                    agg_layer_list = []
                    for clnt , layer in enumerate(l):
                        if(torch.any(label_split[clnt] == i)):
                            agg_layer_list.append(layer[i])
                    if len(agg_layer_list) != 0:
                        last_weight_layer[i] = (aggregate_layer_weight(agg_layer_list))
                        # pad with prev.
            elif(i == num_layers):
                store_prev = {}
                
                for k in missing_labels:
                    k = k.item()
                    store_prev[k] = global_values[-1][k]
                
                global_values[-1] = torch.from_numpy((aggregate_last_layer(l , label_split)))
                
                for k in missing_labels:
                    k = k.item()
                    global_values[-1][k] = store_prev[k]
            else:
                agg_layers.append(aggregate_layer(l))
        


        # keep the rest of global parameters as it is
        for i , layer in enumerate(agg_layers):
            layer = torch.from_numpy(layer)
            if layer.dim() == 0:
                global_values[i] = layer
            else:
                layer_shape = layer.shape
                slices = [slice(0, dim) for dim in layer_shape]
                global_values[i][slices] = layer
        
        for i , inner_layer in enumerate(last_weight_layer):

            if(inner_layer is None):
                continue
            
            inner_layer = torch.from_numpy(inner_layer)
            inner_layer_shape = inner_layer.shape
            slices = [slice(0, dim) for dim in inner_layer_shape]
            global_values[num_layers - 2][i][:inner_layer_shape[0]] = inner_layer

        
        new_state_dict = {}
        for i , k in enumerate(self.model.state_dict().keys()):
            new_state_dict[k] = global_values[i]
        self.model.load_state_dict(new_state_dict)
        

        for i in range(len(global_values)):
            global_values[i] = global_values[i].numpy()


        return ndarrays_to_parameters(global_values) , {}

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