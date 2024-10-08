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
    def __init__(self, power_domain_api: PowerDomainApi, client_load_api: ClientLoadApi, scenario: Scenario, cfg: DictConfig, client_to_batches, client_labels) -> None:
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
        self.client_labels = client_labels
        self.all_labels = set(element for sublist in self.client_labels for element in sublist)
        # all_clients = self.client_load_api.get_clients()

        self.time_now = None
        self.total_carbon_foorprint = 0
        self.client_to_batches = client_to_batches
        self.carbon_foot_print_history = {}
        # self.client_history = {client : {'weighted_p_c' : 0, } for client in all_clients}
        # self._clients_to_cid = clients_to_cid
        self.total_time_taken = 0


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

    # def all(self) -> Dict[str, ClientProxy]:
    #     """Return all available clients."""
    #     return self.clients

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        all_clients = self.client_load_api.get_clients()
        clnts_as_np_clnts = []
        for clnt in all_clients:
            clnts_as_np_clnts.append(self.clients[clnt.name])
        return random.sample(clnts_as_np_clnts, 10)
        # return clnts_as_np_clnts

    def sample(
        self,
        num_clients: int,
        server_round: int,
        prev_rnd_clnts_stat_util: List[Tuple[str, float]], 
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        get_all_available_clients = False
    ) -> List[ClientProxy]:
        
        if server_round > 0:
            all_clients_dict = self.client_load_api.get_clients_as_dict()
            for i, stat_util in prev_rnd_clnts_stat_util:
                all_clients_dict[i].record_statistical_utility(server_round - 1, stat_util)


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
            self.cycle_participation_mean = np.mean([c.weighted_participated_rounds for c in self.cycle_active_clients])
            self.cycle_active_clients = set()
            print(f"############################################################")
            print(f"### NEW CYCLE! MEAN: {self.cycle_participation_mean} ###")
            print(f"############################################################")
        elif self.cycle_start + timedelta(hours=24 - TRANSITION_PERIOD_H) <= self.time_now:
            current_mean = np.mean([c.weighted_participated_rounds for c in self.cycle_active_clients])
            factor = (self.time_now - (self.cycle_start + timedelta(hours=24 - TRANSITION_PERIOD_H))).seconds / 3600 / TRANSITION_PERIOD_H
            wallah = self.cycle_participation_mean + (current_mean - self.cycle_participation_mean) * factor
            print(f"Cycle mean: {self.cycle_participation_mean:.2f}, Current mean: {current_mean:.2f} factor: {factor}, result: {wallah} ###")


        clnts = _filterby_current_capacity_and_energy(self.power_domain_api, self.client_load_api, self.time_now, self.cfg)

        # myclients = sorted(clnts, key=_sort_key, reverse=True)
        self.cycle_active_clients.union(clnts)
        
        self._update_excluded_clients(clnts, server_round, wallah)

        clnts = [client for client in clnts if client not in self.excluded_clients]

        i = 0
        while(True):
            filtered_clients = _filterby_forecasted_capacity_and_energy(self.power_domain_api, self.client_load_api, clnts, self.time_now, self.cfg, duration=i)
            i += 1
            # loop to find out atleast two 1s
            client_classes = [_batches_to_class(i, self.client_to_batches[int(clnt.name.split('_')[0])] * self.cfg.Simulation.EPOCHS) for clnt, i in filtered_clients]
            if client_classes.count(1) >= 2:
                break
        print(f'i = {i}')
        self.total_time_taken += i
        # for i, (c, _) in enumerate(filtered_clients):
        #     filtered_clients[i] = (c, client_classes[i])
        
        # filtered_clients = sorted(filtered_clients, key=_sort_key, reverse=True)

        num_clients_to_sample = 10
        all_classes = {1:[], 0.5:[], 0.25:[], 0.125:[], 0.0625: []}
        temp_classes = []
        for k in all_classes.keys():
            temp_classes.extend([k for _ in range(int(num_clients_to_sample / len(all_classes)))])
        
        sampled_clients = []
        covered_labels = set()
        
        for c, b in filtered_clients:
            # if covered labels is already equal to all labels then u could directly add the one
            if len(covered_labels) == len(self.all_labels):
                sampled_clients.append((c, b))
                continue
            if set(self.client_labels[int(c.name.split('_')[0])]).issubset(covered_labels):
                print('# eat five star do nothing')
            else:
                sampled_clients.append((c, b))
            # if clients labels are already present in covered labels, no need to add that
            
        filtered_clients = sampled_clients
        class_assignment = [min(a[1], b) for a, b in zip(filtered_clients[:num_clients_to_sample], temp_classes)]
        filtered_clients = filtered_clients[:num_clients_to_sample]
        for i, (cl, bts) in enumerate(filtered_clients):
            filtered_clients[i] = (cl, class_assignment[i])

        # for i, cls in enumerate(client_classes):
        #     all_classes[cls].append(i)


        # sampled_filtered_clients = []
        # remember_the_index = []
        # num_clients_sampled = 0
        # for k in all_classes.keys():
        #     if len(all_classes[k]) != 0:
        #         # indices = random.sample(all_classes[k], 2) if len(all_classes[k]) >= 2 else [all_classes[k][0]]
        #         indices = all_classes[k][:2] if len(all_classes[k]) >= 2 else [all_classes[k][0]]
        #         num_clients_sampled += 2 if len(all_classes[k]) >= 2 else 1
        #         for index in indices:
        #             sampled_filtered_clients.append(filtered_clients[index])
        #             remember_the_index.append(index)

        # remember_the_index = sorted(remember_the_index, reverse=True)
        # if len(sampled_filtered_clients) < min_num_clients:
        #     for index_already_used in remember_the_index:
        #         filtered_clients.pop(index_already_used)
        #         sampled_filtered_clients.extend(random.sample(filtered_clients, 2))
        # filtered_clients = sampled_filtered_clients

        self.time_now += timedelta(minutes=random.randint(10, 60))
        
        # filtered_clients = filtered_clients[:num_clients]
        carbon_footprint_till_now = 0

        for client, model_rate in filtered_clients:
            batches_in_client = self.client_to_batches[int(client.name.split('_')[0])] * self.cfg.Simulation.EPOCHS
            carbon_footprint_till_now += client.record_usage( batches_in_client, model_rate)


        this_round_carbon_footprint = _ws_to_kwh(carbon_footprint_till_now)
        self.total_carbon_foorprint += this_round_carbon_footprint
        #carbon footprint history
        self.carbon_foot_print_history[server_round] = this_round_carbon_footprint

        # print('testing carbon footprint = ', _ws_to_kwh(sum(client.participated_batches * client.energy_per_batch for client in self.client_load_api.get_clients())))
        print(f"Renewable excess energy consumed till now ({server_round}) = {self.total_carbon_foorprint} kwh")
        #print the carbon footprint history
        print(f"Renewable excess energy consumption history = ", self.carbon_foot_print_history)
        # print(f'total_time_taken = {self.total_time_taken}')

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
        for clnt, model_rate in clients:
            cids.append((self.clients[clnt.name], model_rate))
        return cids
    

    def _update_excluded_clients(self, clients: List[Client], round_number, wallah: float) -> Tuple[List[Tuple[Client, float]]]:
        alpha = self.cfg.client_selection.alpha
        exclusion_factor = self.cfg.client_selection.exclusion_factor
        rng = np.random.default_rng(seed=self.cfg.Scenario.seed)

        # for client in clients:
        #     print(f'client name = {client.name}', client.participated_in_last_round(round_number))
        participants = {client for client in clients if client.participated_in_last_round(round_number)}

        if not participants:
            return clients
        print(f"| Participants: {len(participants)}")

        # Calculate utility threshold
        utility_threshold = np.quantile([client.statistical_utility() for client in participants], exclusion_factor)
        print(f"| Utility threshold: {utility_threshold:.2f} (quantile {exclusion_factor})")

        # Exclude clients below threshold
        newly_excluded = [client for client in participants if client.statistical_utility() <= utility_threshold]
        self.excluded_clients.extend(newly_excluded)

        print(f"| Excluded clients after add: {len(self.excluded_clients)}")
        for i, client in enumerate(self.excluded_clients):
            participated_rounds = client.weighted_participated_rounds - wallah
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
        with open("filtered_clients.txt", "a") as f:
            f.write(f'current capacity Clients = {str(clients)}\n')
            # f.write(f"Is_participated = {is_participated}\n")
            f.write(f"Participants in current round  = {participants}\n")
            f.write(f"number of excluded clients {len(self.excluded_clients)}\n")
            f.write(f"Excluded clients: {self.excluded_clients}\n")

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
                                             now: datetime, cfg: DictConfig, duration = _DURATION) -> List[Tuple[Client, float]]:
    # print("Manasa nuvvu unde chote cheppamma")
    # print(clients)
    # print("\n\n")
    filtered_clients: List[Tuple[Client, float]] = []
    to_print = []

    for client in clients:
        possible_batches = client_load_api.forecast(now, client_name=client.name, duration_in_timesteps=duration, cfg=cfg)
        ree_powered_batches = power_domain_api.forecast(start_time=now, zone=client.zone, duration_in_timesteps=duration, cfg=cfg) / client.energy_per_batch
        total_max_batches = np.minimum(possible_batches.values, ree_powered_batches.values).sum()
        # if total_max_batches >= client.batches_per_epoch * min_epochs:
        filtered_clients.append((client, total_max_batches))
        # to_select, batches_if_selected = _has_more_resources_in_future(possible_batches, ree_powered_batches)
        # if not to_select:
        #     filtered_clients.append((client, batches_if_selected))
        #     to_print.append(client)
    filtered_clients = sorted(filtered_clients, key=_sort_key, reverse=True)
    
    return filtered_clients


def _sort_key(client):
    # return client.batches_per_timestep * client.energy_per_batch
    return client[1]

def _batches_to_class(batches, client_batches_to_execute):
    # return 1
    model_size = 1
    for _ in range(5):
        if batches >= client_batches_to_execute * model_size:
            return model_size
        model_size /= 2
    # if batches <= 10:
    #     return 0.0625
    # elif batches <= 20:
    #     return 0.125
    # elif batches <= 30:
    #     return 0.25
    # elif batches <= 40:
    #     return 0.5
    # else:
    #     return 1
    return 0.0625

def _ws_to_kwh(ws: float) -> float:
    return ws / 3600 / 1000