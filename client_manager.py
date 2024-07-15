# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower ClientManager."""


import random
import threading
from logging import INFO
from typing import Dict, List, Optional, Tuple
import copy
import numpy as np

import flwr as fl
import torch
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from entities import PowerDomainApi, ClientLoadApi, Client
from datetime import datetime, timedelta
from scenarios import Scenario
from omegaconf import DictConfig

_DURATION = 5

class FedZeroCM(fl.server.ClientManager):
    def __init__(self, power_domain_api: PowerDomainApi, client_load_api: ClientLoadApi, scenario: Scenario, cfg: DictConfig) -> None:
        self.clients: Dict[str, ClientProxy] = {}
        self._cv = threading.Condition()
        self.power_domain_api = power_domain_api
        self.client_load_api = client_load_api
        self.scenario = scenario
        self.cfg = cfg
        self.excluded_clients = []
        self.rng = np.random.default_rng(seed= cfg.Scenario.seed)
        self.client_selection_history = {}
        self.cycle_start = None
        self.cycle_active_clients = set()
        self.cycle_participation_mean = 0

        # all_clients = self.client_load_api.get_clients()

        self.time_now = None
        # self.client_history = {client : {'weighted_p_c' : 0, } for client in all_clients}
        # self._clients_to_cid = clients_to_cid


    def __len__(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self.clients)

    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are available.

        Blocks until the requested number of clients is available or until a
        timeout is reached. Current timeout default: 1 day.

        Parameters
        ----------
        num_clients : int
            The number of clients to wait for.
        timeout : int
            The time in seconds to wait for, defaults to 86400 (24h).

        Returns
        -------
        success : bool
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        if client.cid in self.clients:
            del self.clients[client.cid]

            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients


    def sample(
        self,
        num_clients: int,
        server_round: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        if min_num_clients is None:
            min_num_clients = num_clients

        if self.time_now is None:
            self.time_now = self.scenario.start_date

        TRANSITION_PERIOD_H = 12
        wallah = self.cycle_participation_mean
        if self.cycle_start is None:
            self.cycle_start = self.time_now
        elif self.cycle_start + timedelta(hours=24) <= self.time_now:
            self.cycle_start = self.time_now
            self.cycle_participation_mean = np.mean([c.participated_rounds for c in self.cycle_active_clients])
            self.cycle_active_clients = set()
            print(f"############################################################")
            print(f"### NEW CYCLE! MEAN: {self.cycle_participation_mean} ###")
            print(f"############################################################")
        elif self.cycle_start + timedelta(hours=24 - TRANSITION_PERIOD_H) <= self.time_now:
            current_mean = np.mean([c.participated_rounds for c in self.cycle_active_clients])
            factor = (self.time_now - (self.cycle_start + timedelta(hours=24 - TRANSITION_PERIOD_H))).seconds / 3600 / TRANSITION_PERIOD_H
            wallah = self.cycle_participation_mean + (current_mean - self.cycle_participation_mean) * factor
            print(f"Cycle mean: {self.cycle_participation_mean:.2f}, Current mean: {current_mean:.2f} factor: {factor}, result: {wallah} ###")


        clnts = _filterby_current_capacity_and_energy(self.power_domain_api, self.client_load_api, self.time_now, self.cfg)

        # myclients = sorted(clnts, key=_sort_key, reverse=True)
        self.cycle_active_clients.union(clnts)
        
        self._update_excluded_clients(clnts, server_round, wallah)

        clnts = [client for client in clnts if client not in self.excluded_clients]

        filtered_clients = _filterby_forecasted_capacity_and_energy(self.power_domain_api, self.client_load_api, clnts, self.time_now, self.cfg)
        
        self.time_now += timedelta(minutes=15)
        
        for client, batches_to_compute in filtered_clients:
            client.record_usage(batches_to_compute, _batches_to_class(batches_to_compute))
            client.record_statistical_utility(server_round, 1000)

        cids_filtered_clients = self._clients_to_numpy_clients(filtered_clients)

        filtered_client_proxies = []
        for cpr, model_size in cids_filtered_clients:
            cpr.properties['model_rate'] = model_size
            filtered_client_proxies.append(cpr)
        
        selected_clients = filtered_client_proxies[:num_clients]
        if (len(selected_clients) >= self.cfg.Simulation.CLIENTS_PER_ROUND):
            selected_clients = selected_clients[:self.cfg.Simulation.CLIENTS_PER_ROUND]

        # Update selection history
        for client in selected_clients:
            if client.cid not in self.client_selection_history:
                self.client_selection_history[client.cid] = []
            self.client_selection_history[client.cid].append(server_round)

        # Print selection information
        print(f"\n--- Round {server_round} ---")
        print("Selected clients:")
        for client in selected_clients:
            selection_count = len(self.client_selection_history[client.cid])
            print(f"  - Client {client.cid}: selected {selection_count} times, rounds {self.client_selection_history[client.cid]}")

        print("\nExcluded clients:")
        for client_name in self.excluded_clients:
            print(f"  - Client {client_name}")

        print("\nSelection summary:")
        print(f"  Total clients available: {len(clnts)}")
        print(f"  Clients after forecasting: {len(filtered_clients)}")
        print(f"  Clients selected: {len(selected_clients)}")
        print(f"  Clients excluded: {len(self.excluded_clients)}")

        return selected_clients

    def _clients_to_numpy_clients(self, clients):
        cids = []
        for clnt, batches in clients:
            cids.append((self.clients[clnt.name], _batches_to_class(batches)))
        return cids
    

    def _update_excluded_clients(self, clients: List[Client], round_number, wallah: float) -> Tuple[List[Tuple[Client, float]]]:
        alpha = self.cfg.client_selection.alpha
        exclusion_factor = self.cfg.client_selection.exclusion_factor
        rng = np.random.default_rng(seed=self.cfg.Scenario.seed)

        # participants = [client for client in clients if client.name not in excluded_clients]
        participants = {client for client in clients if client.participated_in_last_round(round_number)}

        if not participants:
            return clients

        # Calculate utility threshold
        utility_threshold = np.quantile([client.statistical_utility() for client in participants], exclusion_factor)

        # Exclude clients below threshold
        newly_excluded = [client for client in participants if client.statistical_utility() <= utility_threshold]
        self.excluded_clients.extend(newly_excluded)

        print(f"| Excluded clients after add: {len(self.excluded_clients)}")
        for i, client in enumerate(self.excluded_clients):
            participated_rounds = client.participated_rounds - wallah
            if participated_rounds > 0:
                probability = min(alpha * 1 / participated_rounds, 1)
            else:
                probability = 1

            print(f"| #{i} {client.name}: {client.participated_rounds} part -> {probability:.0%} ...", end="")
            if rng.random() <= probability:
                print(" SUCCESS")
                if client in self.excluded_clients:
                    self.excluded_clients.remove(client)
            else:
                print("")
        print(f"| Excluded clients after remove: {len(self.excluded_clients)}")
        print("----------------------------------------------")

        # Filter clients based on updated exclusion list
        # updated_clients = [client for client in clients if client.name not in self.excluded_clients]

        # return updated_clients

def _filterby_current_capacity_and_energy(power_domain_api: PowerDomainApi,
                                          client_load_api: ClientLoadApi,
                                          now: datetime, cfg: DictConfig
                                          ) -> List[Client]:
    zones_with_energy = [zone for zone in power_domain_api.zones if power_domain_api.actual(now, zone, cfg) > 0.0]
    clients = [client for client in client_load_api.get_clients(zones_with_energy) if client_load_api.actual(now, client.name) > 0.0]
    print(f"There are {len(clients)} clients available across {len(zones_with_energy)} power domains.")
    return clients

def _filterby_forecasted_capacity_and_energy(power_domain_api: PowerDomainApi,
                                             client_load_api: ClientLoadApi,
                                             clients: List[Client],
                                             now: datetime, cfg: DictConfig) -> List[Tuple[Client, float]]:
    filtered_clients: List[Tuple[Client, float]] = []
    for client in clients:
        possible_batches = client_load_api.forecast(now, client_name=client.name, duration_in_timesteps=_DURATION, cfg=cfg)
        ree_powered_batches = power_domain_api.forecast(start_time=now, zone=client.zone, duration_in_timesteps=_DURATION, cfg=cfg) / client.energy_per_batch
        to_select, batches_if_selected = _has_more_resources_in_future(possible_batches, ree_powered_batches)
        if not to_select:
            filtered_clients.append((client, batches_if_selected))
    filtered_clients = sorted(filtered_clients, key=_sort_key, reverse=True)
    return filtered_clients

def _sort_key(client):
    # return client.batches_per_timestep * client.energy_per_batch
    return client[1]

def _has_more_resources_in_future(possible_batches, ree_powered_batches):
    total_max_batches = np.max(np.minimum(possible_batches.values, ree_powered_batches.values))
    batches_if_selected = min(possible_batches.to_list()[0], ree_powered_batches.to_list()[0])
    return (False, batches_if_selected) if (total_max_batches == batches_if_selected) else (True, 0)

def _batches_to_class(batches):
    if batches <= 10:
        return 0.0625
    elif batches <= 20:
        return 0.125
    elif batches <= 30:
        return 0.25
    elif batches <= 40:
        return 0.5
    else:
        return 1

