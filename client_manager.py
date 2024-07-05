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
from typing import Dict, List, Optional
import copy

import flwr as fl
import torch
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from entities import PowerDomainApi, ClientLoadApi, Client
from datetime import datetime, timedelta
from scenarios import Scenario
from omegaconf import DictConfig

import numpy as np

# Change cheyya rareiiii
_DURATION = 5

class FedZeroCM(fl.server.ClientManager):
    """Provides a pool of available clients."""

    def __init__(self, power_domain_api : PowerDomainApi, client_load_api : ClientLoadApi, scenario: Scenario, cfg: DictConfig)-> None:
        self.clients: Dict[str, ClientProxy] = {}
        self._cv = threading.Condition()
        self.power_domain_api = power_domain_api
        self.client_load_api = client_load_api
        self.scenario = scenario
        self.cfg = cfg
        self.excluded_clients = []
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
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients


        time_now = timedelta(minutes = server_round * 5) + self.scenario.start_date
        clnts = _filterby_current_capacity_and_energy(self.power_domain_api, self.client_load_api, time_now, self.cfg)

        # tharavatha excluded clients ni tesesi RAREIII
        # clnts = [client for client in clnts if client not in self.excluded_clients]

        
        # Kattappa eeda code marchalne
        # Shall change the sort function accordingly
        myclients = sorted(clnts, key = _sort_key, reverse = True)

        filtered_clients = _filterby_forecasted_capacity_and_energy(self.power_domain_api, self.client_load_api, myclients, time_now, self.cfg)
        filtered_clients = _update_excluded_clients(filtered_clients, self.excluded_clients, self.cfg)
        self.excluded_clients = copy.deep_copy(filtered_clients)
        cids_filtered_clients = self._clients_to_numpy_clients(filtered_clients)


        filtered_client_proxies = []
        for cpr, model_size in cids_filtered_clients:
            cpr.properties['model_rate'] = model_size
            filtered_client_proxies.append(cpr)
        
        # return [self.clients[cid] for cid, model_size in cids_filtered_clients]
        return filtered_client_proxies

        # self.wait_for(min_num_clients)
        # # Sample clients which meet the criterion
        # available_cids = list(self.clients)
        # if criterion is not None:
        #     available_cids = [
        #         cid for cid in available_cids if criterion.select(self.clients[cid])
        #     ]

        # if num_clients > len(available_cids):
        #     log(
        #         INFO,
        #         "Sampling failed: number of available clients"
        #         " (%s) is less than number of requested clients (%s).",
        #         len(available_cids),
        #         num_clients,
        #     )
        #     return []

        # sampled_cids = random.sample(available_cids, num_clients)
        # return [self.clients[cid] for cid in sampled_cids]

    def _clients_to_numpy_clients(self, clients):
        # print("Gemini Ganesan client manager clients to numpy clients")
        # print('length = ', len(self.clients))
        cids = []
        for clnt, batches in clients:
            cids.append((self.clients[clnt.name], _batches_to_class(batches)))
        return cids




def _filterby_current_capacity_and_energy(power_domain_api: PowerDomainApi,
                                          client_load_api: ClientLoadApi,
                                          now : datetime, cfg:DictConfig
                                          ) -> List[Client]:
    zones_with_energy = [zone for zone in power_domain_api.zones if power_domain_api.actual(now, zone, cfg) > 0.0]
    clients = [client for client in client_load_api.get_clients(zones_with_energy) if client_load_api.actual(now, client.name) > 0.0]
    print(f"There are {len(clients)} clients available across {len(zones_with_energy)} power domains.")
    return clients


def _filterby_forecasted_capacity_and_energy(power_domain_api: PowerDomainApi,
                                             client_load_api: ClientLoadApi,
                                             clients: List[Client],
                                             now: datetime, cfg:DictConfig) -> List[Client]:
    filtered_clients: List[Client] = []
    for client in clients:
        # print('error ikkada ?')
        possible_batches = client_load_api.forecast(now, client_name=client.name, duration_in_timesteps=_DURATION, cfg=cfg)
        # print('possible batches')
        # print(possible_batches.to_list())

        # print('leka pothe ikkada? ')
        ree_powered_batches = power_domain_api.forecast(start_time=now, zone=client.zone, duration_in_timesteps=_DURATION, cfg=cfg) / client.energy_per_batch
        # print('ree_powered_batches')
        # print(ree_powered_batches.to_list())
        # # Significantly faster than pandas
        to_select, batches_if_selected = _has_more_resources_in_future(possible_batches, ree_powered_batches)
        if to_select:
            print('not adding')
        else:
            # print('adding')
            filtered_clients.append((client, batches_if_selected))
        print('\n\n')
        # if total_max_batches >= client.batches_per_epoch(cfg) * min_epochs:
        #     filtered_clients.append(client)
    # print(len(filtered_clients))
    # print(filtered_clients)

    return filtered_clients


def _sort_key(client):
    return client.batches_per_timestep * client.energy_per_batch


def _has_more_resources_in_future(possible_batches, ree_powered_batches):
    # print('minimum ')
    total_max_batches = np.max(np.minimum(possible_batches.values, ree_powered_batches.values))
    
    batches_if_selected = min(possible_batches.to_list()[0], ree_powered_batches.to_list()[0])
    # print('total max batches')
    # print(total_max_batches)
    return (False,batches_if_selected) if (total_max_batches == batches_if_selected) else (True, 0)

def _batches_to_class(batches):
    # categorise the batches into classes based on number of batches
    #print all the batches
    # print('batches')
    print(batches)
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



def _update_excluded_clients(clients, excluded_clients, cfg):
    updated_clients = []
    for client in clients:
        if client not in excluded_clients:
            updated_clients.append(client)
    return updated_clients



    