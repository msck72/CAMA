import functools
import json
import multiprocessing
import os.path
import random
from typing import Dict, List, Tuple
import urllib
import zipfile

import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as transforms
from scipy.special import softmax
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, Subset
from tqdm import tqdm
from omegaconf import DictConfig

# from config import NIID_DATA_SEED
from kwt.utils.dataset import get_loader

ALL_LETTERS = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def get_dataloaders(dataset: str, num_clients: int, batch_size: int, beta: float, cfg: DictConfig):
    np.random.seed(cfg.Simulation['NIID_DATA_SEED'])
    random.seed(cfg.Simulation['NIID_DATA_SEED'])
    torch.manual_seed(cfg.Simulation['NIID_DATA_SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.Simulation['NIID_DATA_SEED'])
    print(f'NIID data seed: {cfg.Simulation["NIID_DATA_SEED"]}')

    if dataset == 'mnist':
        trainloaders, testloader = load_mnist(num_clients, batch_size, beta)
        num_classes = len(np.unique(testloader.dataset.targets))
    elif dataset in ['cifar10', 'cifar100']:
        # trainloaders, testloader = load_cifar(dataset.upper(), num_clients, batch_size, beta)
        # num_classes = len(np.unique(testloader.dataset.targets))
        num_classes = 10
        trainloaders, testloader = load_cifar10_based_on_classes_per_client(num_clients, cfg.Scenario.shared_per_user , num_classes , batch_size, seed=cfg.Simulation['NIID_DATA_SEED'])
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not implemented")
    return trainloaders, testloader, num_classes


def load_cifar(cifar_type: str, num_clients: int, batch_size: int, beta: float):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Mean and Standard deviation of CIFAR10: https://github.com/kuangliu/pytorch-cifar/issues/19
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    trainset = getattr(torchvision.datasets, cifar_type)(
        "./data", train=True, download=True, transform=train_transforms
    )
    testset = getattr(torchvision.datasets, cifar_type)(
        "./data", train=False, download=True, transform=test_transforms
    )
    trainloaders = []
    if 0.0 < beta < 1.0:
        client_to_data_ids = _get_niid_client_data_ids(trainset, num_clients, beta)
        for client_id in client_to_data_ids:
            tmp_client_img_ids = client_to_data_ids[client_id]
            tmp_train_sampler = SubsetRandomSampler(tmp_client_img_ids)
            _append_to_dataloaders(trainset, batch_size, trainloaders, tmp_train_sampler)
    else:
        partition_size = len(trainset) // num_clients
        lengths = [partition_size] * num_clients
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
        for dataset in datasets:
            _append_to_dataloaders(dataset, batch_size, trainloaders)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloaders, testloader


def load_cifar10_based_on_classes_per_client(num_clients: int, shared_per_user: int, num_classes: int, batch_size: int, seed = 42):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Mean and Standard deviation of CIFAR10: https://github.com/kuangliu/pytorch-cifar/issues/19
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=train_transforms
    )
    testset = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=test_transforms
    )

    client_trainsets = non_iid(trainset, num_classes, num_clients, shared_per_user, seed=seed)
    client_train_data_loaders = []

    for dataset in client_trainsets:
        client_train_data_loaders.append(DataLoader(dataset,batch_size=batch_size,shuffle=True))
    
    return client_train_data_loaders, DataLoader(testset, batch_size=batch_size, shuffle=False)

def load_mnist(num_clients: int, batch_size: int, beta: float):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=train_transforms
    )
    testset = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=test_transforms
    )
    trainloaders = []
    if 0.0 < beta < 1.0:
        client_to_data_ids = _get_niid_client_data_ids(trainset, num_clients, beta)
        for client_id in client_to_data_ids:
            tmp_client_img_ids = client_to_data_ids[client_id]
            tmp_train_sampler = SubsetRandomSampler(tmp_client_img_ids)
            _append_to_dataloaders(trainset, batch_size, trainloaders, tmp_train_sampler)
    else:
        partition_size = len(trainset) // num_clients
        lengths = [partition_size] * num_clients
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
        for dataset in datasets:
            _append_to_dataloaders(dataset, batch_size, trainloaders)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloaders, testloader 



def _append_to_dataloaders(trainset, batch_size, trainloaders, random_sampler=None):
    if random_sampler is None:
        trainloaders.append(DataLoader(trainset, batch_size=batch_size, shuffle=True))
    else:
        trainloaders.append(DataLoader(trainset, batch_size=batch_size, sampler=random_sampler))


def _get_niid_client_data_ids(dataset: Dataset, num_clients: int, beta: float):
    labels = np.array(dataset.targets)
    client_to_data_ids = {k: [] for k in range(num_clients)}
    for label_id in range(len(np.unique(labels))):
        idx_batch = [[] for _ in range(num_clients)]
        label_ids = np.where(labels == label_id)[0]
        label_proportions = np.random.dirichlet(np.repeat(beta, num_clients))
        label_proportions = np.cumsum(label_proportions * len(label_ids)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(label_ids, label_proportions))]
        for client_id in range(num_clients):
            client_to_data_ids[client_id] += idx_batch[client_id]
    return client_to_data_ids


def non_iid(
    dataset,
    classes_size,
    num_clients: int,
    shard_per_user: int,
    label_split=None,
    seed=42,
) -> Tuple[List[Dataset], List]:

    data_split: Dict[int, List] = {i: [] for i in range(num_clients)}

    label_idx_split, shard_per_class = _split_dataset_targets_idx(
        dataset,
        shard_per_user,
        num_clients,
        classes_size,
    )

    if label_split is None:
        label_split = list(range(classes_size)) * shard_per_class
        label_split = torch.tensor(label_split)[
            torch.randperm(
                len(label_split), generator=torch.Generator().manual_seed(seed)
            )
        ].tolist()
        label_split = np.array(label_split).reshape((num_clients, -1)).tolist()

        for i, _ in enumerate(label_split):
            label_split[i] = np.unique(label_split[i]).tolist()

    for i in range(num_clients):
        for label_i in label_split[i]:
            idx = torch.arange(len(label_idx_split[label_i]))[
                torch.randperm(
                    len(label_idx_split[label_i]),
                    generator=torch.Generator().manual_seed(seed),
                )[0]
            ].item()
            data_split[i].extend(label_idx_split[label_i].pop(idx))

    return _get_dataset_from_idx(dataset, data_split, num_clients)


def _split_dataset_targets_idx(dataset, shard_per_user, num_clients, classes_size):
    label = np.array(dataset.target) if hasattr(dataset, "target") else dataset.targets
    label_idx_split: Dict = {}
    for i, _ in enumerate(label):
        label_i = label[i]
        if label_i not in label_idx_split:
            label_idx_split[label_i] = []
        label_idx_split[label_i].append(i)

    shard_per_class = int(shard_per_user * num_clients / classes_size)

    for label_i in label_idx_split:
        label_idx = label_idx_split[label_i]
        num_leftover = len(label_idx) % shard_per_class
        leftover = label_idx[-num_leftover:] if num_leftover > 0 else []
        new_label_idx = (
            np.array(label_idx[:-num_leftover])
            if num_leftover > 0
            else np.array(label_idx)
        )
        new_label_idx = new_label_idx.reshape((shard_per_class, -1)).tolist()

        for i, leftover_label_idx in enumerate(leftover):
            new_label_idx[i] = np.concatenate([new_label_idx[i], [leftover_label_idx]])
        label_idx_split[label_i] = new_label_idx
    return label_idx_split, shard_per_class




def _get_dataset_from_idx(dataset, data_split, num_clients):
    divided_dataset = [None for i in range(num_clients)]
    for i in range(num_clients):
        divided_dataset[i] = Subset(dataset, data_split[i])
    return divided_dataset

