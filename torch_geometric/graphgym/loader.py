import os.path as osp
from typing import Callable

import logging
import torch

import torch_geometric.graphgym.register as register
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    PPI,
    Amazon,
    Coauthor,
    KarateClub,
    MNISTSuperpixels,
    Planetoid,
    QM7b,
    TUDataset,
)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.transform import (
    create_link_label,
    neg_sampling_transform,
)
from torch_geometric.loader import (
    ClusterLoader,
    DataLoader,
    LinkNeighborLoader,
    NeighborLoader,
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
    RandomNodeLoader,
)
from torch_geometric.sampler import NegativeSampling
from torch_geometric.utils import (
    index_to_mask,
    negative_sampling,
    to_undirected,
)

index2mask = index_to_mask  # TODO Backward compatibility


def planetoid_dataset(name: str) -> Callable:
    return lambda root: Planetoid(root, name)


register.register_dataset('Cora', planetoid_dataset('Cora'))
register.register_dataset('CiteSeer', planetoid_dataset('CiteSeer'))
register.register_dataset('PubMed', planetoid_dataset('PubMed'))
register.register_dataset('PPI', PPI)


def load_pyg(name, dataset_dir):
    """Load PyG dataset objects. (More PyG datasets will be supported).

    Args:
        name (str): dataset name
        dataset_dir (str): data directory

    Returns: PyG dataset object

    """
    dataset_dir = osp.join(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset = TUDataset(dataset_dir, name, transform=T.Constant())
        else:
            dataset = TUDataset(dataset_dir, name[3:])
    elif name == 'Karate':
        dataset = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset = Coauthor(dataset_dir, name='CS')
        else:
            dataset = Coauthor(dataset_dir, name='Physics')
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset = Amazon(dataset_dir, name='Computers')
        else:
            dataset = Amazon(dataset_dir, name='Photo')
    elif name == 'MNIST':
        dataset = MNISTSuperpixels(dataset_dir)
    elif name == 'PPI':
        dataset = PPI(dataset_dir)
    elif name == 'QM7b':
        dataset = QM7b(dataset_dir)
    else:
        raise ValueError(f"'{name}' not support")

    return dataset


def set_dataset_attr(dataset, name, value, size):
    dataset._data_list = None
    dataset.data[name] = value
    if dataset.slices is not None:
        dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)


def load_ogb(name, dataset_dir):
    r"""Load OGB dataset objects.

    Args:
        name (str): dataset name
        dataset_dir (str): data directory

    Returns: PyG dataset object

    """
    from ogb.graphproppred import PygGraphPropPredDataset
    from ogb.linkproppred import PygLinkPropPredDataset
    from ogb.nodeproppred import PygNodePropPredDataset

    if name[:4] == 'ogbn':
        # ogbn-proteins doesn't have node features
        if name == 'ogbn-proteins':
            dataset = PygNodePropPredDataset(name=name, root=dataset_dir, transform=T.Constant())
        else:
            dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = ['train_mask', 'val_mask', 'test_mask']
        for i, key in enumerate(splits.keys()):
            # consider only paper-paper relations for ogbn-mag
            if name == 'ogbn-mag':
                mask = index_to_mask(splits[key]['paper'], size=dataset._data.y_dict['paper'].shape[0])
                set_dataset_attr(dataset, split_names[i], mask, len(mask))
            else:
                mask = index_to_mask(splits[key], size=dataset._data.y.shape[0])
                set_dataset_attr(dataset, split_names[i], mask, len(mask))
        if name == 'ogbn-mag':
            # consider only paper-paper edge index for ogbn-mag
            edge_index = to_undirected(dataset._data.edge_index_dict[('paper', 'cites', 'paper')])
            # add node attributes and labels
            set_dataset_attr(dataset, 'x', dataset.x_dict['paper'], dataset.x_dict['paper'][0])
            set_dataset_attr(dataset, 'y', dataset.y_dict['paper'], dataset.y_dict['paper'][0])
        else:
            edge_index = to_undirected(dataset._data.edge_index)
        set_dataset_attr(dataset, 'edge_index', edge_index,
                         edge_index.shape[1])

    elif name[:4] == 'ogbg':
        # ogbg-ppa doesn't have node features
        if name == 'ogbg-ppa':
            dataset = PygGraphPropPredDataset(name=name, root=dataset_dir, transform=T.Constant())
        else:
            dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = [
            'train_graph_index', 'val_graph_index', 'test_graph_index'
        ]
        for i, key in enumerate(splits.keys()):
            id = splits[key]
            set_dataset_attr(dataset, split_names[i], id, len(id))

    elif name[:4] == "ogbl":
        # ogbn-ddi doesn't have node features
        if name == 'ogbl-ddi':
            dataset = PygLinkPropPredDataset(name=name, root=dataset_dir, transform=T.Constant())
        else:
            dataset = PygLinkPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_edge_split()
        id = splits['train']['edge'].T
        if cfg.dataset.resample_negative:
            set_dataset_attr(dataset, 'train_pos_edge_index', id, id.shape[1])
            if cfg.train.sampler != 'link_neighbor':
                if name == 'ogbl-ddi':
                    dataset.transform = T.Compose([T.Constant(), neg_sampling_transform])
                else:
                    dataset.transform = neg_sampling_transform
        else:
            if name == 'ogbl-vessel':
                id_neg = splits['train']['edge_neg'].T
            else:
                id_neg = negative_sampling(edge_index=id,
                                           num_nodes=dataset._data.num_nodes,
                                           num_neg_samples=id.shape[1])
            id_all = torch.cat([id, id_neg], dim=-1)
            label = create_link_label(id, id_neg)
            set_dataset_attr(dataset, 'train_edge_index', id_all,
                             id_all.shape[1])
            set_dataset_attr(dataset, 'train_edge_label', label, len(label))

        id, id_neg = splits['valid']['edge'].T, splits['valid']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'val_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'val_edge_label', label, len(label))

        id, id_neg = splits['test']['edge'].T, splits['test']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'test_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'test_edge_label', label, len(label))

    else:
        raise ValueError('OGB dataset: {} non-exist')

    # dtype int to float
    if hasattr(dataset._data, 'x') and name != 'ogbg-molhiv' and name != 'ogbg-molpcba':
        if dataset._data.x is not None and dataset._data.x.dtype == torch.long:
            dataset._data.x = dataset._data.x.type(dtype=torch.float)
    if hasattr(dataset._data, 'edge_attr') and name != 'ogbg-molhiv' and name != 'ogbg-molpcba':
        if dataset._data.edge_attr is not None and dataset._data.edge_attr.dtype == torch.long:
            dataset._data.edge_attr = dataset._data.edge_attr.type(dtype=torch.float)

    if not isinstance(dataset._data.num_nodes, int):
        dataset._data.num_nodes = int(dataset._data.num_nodes)

    return dataset


def load_dataset():
    r"""Load dataset objects.

    Returns: PyG dataset object

    """
    format = cfg.dataset.format
    name = cfg.dataset.name
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    for func in register.loader_dict.values():
        dataset = func(format, name, dataset_dir)
        if dataset is not None:
            return dataset
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        dataset = load_pyg(name, dataset_dir)
    # Load from OGB formatted data
    elif format == 'OGB':
        dataset = load_ogb(name.replace('_', '-'), dataset_dir)
    else:
        raise ValueError(f"Unknown data format '{format}'")
    return dataset


def set_dataset_info(dataset):
    r"""Set global dataset information.

    Args:
        dataset: PyG dataset object

    """
    # get dim_in and dim_out
    try:
        cfg.share.dim_in = dataset._data.x.shape[1]
    except Exception:
        cfg.share.dim_in = 1
    try:
        if cfg.dataset.task_type == 'classification':
            cfg.share.dim_out = torch.unique(dataset._data.y).shape[0]
        else:
            cfg.share.dim_out = dataset._data.y.shape[1]
    except Exception:
        cfg.share.dim_out = 1

    # get edge_dim
    if dataset._data.edge_attr == None:
        cfg.dataset.edge_dim = None
        if cfg.gnn.use_edge_attr:
            cfg.gnn.use_edge_attr = False
            logging.warning("Dataset does not have edge attributes. "
                            "Change gnn.use_edge_attr to False.")
    else:
        cfg.dataset.edge_dim = dataset._data.edge_attr.shape[1]

    # count number of dataset splits
    cfg.share.num_splits = 1
    for key in dataset._data.keys():
        if 'val' in key:
            cfg.share.num_splits += 1
            break
    for key in dataset._data.keys():
        if 'test' in key:
            cfg.share.num_splits += 1
            break


def create_dataset():
    r"""Create dataset object.

    Returns: PyG dataset object

    """
    dataset = load_dataset()
    set_dataset_info(dataset)

    return dataset


def get_loader(dataset, sampler, batch_size, shuffle=True, split='train'):
    pw = cfg.num_workers > 0
    if sampler == "full_batch" or len(dataset) > 1:
        loader_train = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=cfg.num_workers,
                                  pin_memory=True, persistent_workers=pw)
    elif sampler == "neighbor":
        loader_train = NeighborLoader(
            dataset[0], num_neighbors=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
            batch_size=batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, pin_memory=True)
    elif sampler == "random_node":
        loader_train = RandomNodeLoader(dataset[0],
                                        num_parts=cfg.train.train_parts,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True, persistent_workers=pw)
    elif sampler == "saint_rw":
        loader_train = \
            GraphSAINTRandomWalkSampler(dataset[0],
                                        batch_size=batch_size,
                                        walk_length=cfg.train.walk_length,
                                        num_steps=cfg.train.iter_per_epoch,
                                        sample_coverage=0,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True,
                                        persistent_workers=pw)
    elif sampler == "saint_node":
        loader_train = \
            GraphSAINTNodeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  persistent_workers=pw)
    elif sampler == "saint_edge":
        loader_train = \
            GraphSAINTEdgeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  persistent_workers=pw)
    elif sampler == "cluster":
        loader_train = ClusterLoader(
            dataset[0],
            num_parts=cfg.train.train_parts,
            save_dir=osp.join(
                cfg.dataset.dir,
                cfg.dataset.name.replace("-", "_"),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=pw,
        )
    elif sampler == 'link_neighbor':
        if cfg.dataset.format == 'PyG':
            name = cfg.dataset.name
        elif cfg.dataset.format == 'OGB':
            name = cfg.dataset.name.replace('_', '-')
        if name[:4] == "ogbl":
            splits = dataset.get_edge_split()
            if cfg.dataset.resample_negative:
                neg_samp = NegativeSampling(
                    mode='binary',
                    amount=cfg.dataset.edge_negative_sampling_ratio,
                )
                if split == 'train':
                    edge_label_index = splits['train']['edge'].T
                    delattr(dataset.data, 'train_pos_edge_index')
                elif split == 'val':
                    edge_label_index = splits['valid']['edge'].T
                elif split == 'test':
                    edge_label_index = splits['test']['edge'].T
                edge_label = torch.ones(edge_label_index.size(1))
            else:
                neg_samp = None
                if split == 'train':
                    edge_label_index = dataset.data.train_edge_index
                    edge_label = dataset.data.train_edge_label
                    delattr(dataset.data, 'train_edge_index')
                    delattr(dataset.data, 'train_edge_label')
                elif split == 'val':
                    edge_label_index = dataset.data.val_edge_index
                    edge_label = dataset.data.val_edge_label
                    delattr(dataset.data, 'val_edge_index')
                    delattr(dataset.data, 'val_edge_label')
                elif split == 'test':
                    edge_label_index = dataset.data.test_edge_index
                    edge_label = dataset.data.test_edge_label
                    delattr(dataset.data, 'test_edge_index')
                    delattr(dataset.data, 'test_edge_label')
            loader_train = LinkNeighborLoader(
                data=dataset[0],
                num_neighbors=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp] \
                    if split == 'train' else cfg.val.neighbor_sizes[:cfg.gnn.layers_mp],
                edge_label_index=edge_label_index,
                edge_label=edge_label,
                neg_sampling=neg_samp,
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=cfg.num_workers, 
                pin_memory=True,
            )
        else:
            raise NotImplementedError(f"'{sampler}' is not implemented for {cfg.dataset.name}")
    else:
        raise NotImplementedError(f"'{sampler}' is not implemented")

    return loader_train


def create_loader():
    """Create data loader object.

    Returns: List of PyTorch data loaders

    """
    dataset = create_dataset()
    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [
            get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True, split='train')
        ]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            get_loader(dataset, cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True, split='train')
        ]

    # val and test loaders
    for i in range(cfg.share.num_splits - 1):
        split = 'val' if i == 0 else 'test'
        if cfg.dataset.task == 'graph':
            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]]
            loaders.append(
                get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size,
                           shuffle=False, split=split))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(
                get_loader(dataset, cfg.val.sampler, cfg.train.batch_size,
                           shuffle=False, split=split))

    return loaders
