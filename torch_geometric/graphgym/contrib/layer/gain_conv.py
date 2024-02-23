from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value


class GAINConv(MessagePassing):
    r"""
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(
        self,
        nn: Callable,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,  
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        # gain attributes
        self.nn = nn
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        # linear layer for source and target nodes
        self.lin_l = Linear(in_channels, heads * in_channels, bias=True, weight_initializer='glorot')
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * in_channels, bias=True, weight_initializer='glorot')

        # parameter layer post leaky relu that is node independent
        self.att = Parameter(torch.Tensor(1, heads, in_channels))

        # linear layer for edge dimensions
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * in_channels, bias=False, weight_initializer='glorot')
        else:
            self.lin_edge = None

        # no bias
        self.register_parameter('bias', None)

        # attribute for attention score
        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)

    def forward(
            self, 
            x: Union[Tensor, PairTensor], 
            edge_index: Adj,
            edge_attr: OptTensor = None,
            return_attention_weights: bool = None
        ):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # get source and target embeddings
        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            # [shape: num_nodes, in_channels]
            x_l = x
            x_r = x
        else:
            # [shape: num_nodes, in_channels]
            x_l, x_r = x[0], x[1]
        assert x_l is not None
        assert x_r is not None

        # add self-loops
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor) [shape: num_nodes, att_heads, in_channels]
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr, size=None)
        if self.concat:
            # [shape: num_nodes, att_heads * in_channels]
            out = out.view(-1, self.heads * self.in_channels)
        else:
            # [shape: num_nodes, in_channels] 
            out = out.mean(dim=1)
        # [shape: num_nodes, att_heads * in_channels / in_channels] --> [shape: num_nodes, out_channels]
        out = self.nn(out)

        # update alpha to be returned if required
        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out, None

    def message(
            self, 
            x_j: Tensor, 
            x_i: Tensor, 
            edge_attr: OptTensor,
            index: Tensor, 
            ptr: OptTensor,
            size_i: Optional[int]
        ) -> Tensor:
        # pass source and target node embeddings to linear layers
        # [shape: num_edges, in_channels] --> [shape: num_edges, att_heads, in_channels]
        x_j_lin = self.lin_l(x_j).view(-1, self.heads, self.in_channels)
        x_i_lin = self.lin_r(x_i).view(-1, self.heads, self.in_channels)

        # add source and target embeddings to 'emulate' concatentation given that linear layer is applied already
        # [shape: num_edges, att_heads, in_channels]
        x = x_i_lin + x_j_lin

        # check if there are edge attributes
        if edge_attr is not None:
            # check dimensions of edge dimensions
            if edge_attr.dim() == 1:
                # resize if 1
                edge_attr = edge_attr.view(-1, 1)
            # ensure that there is linear layer for edges
            assert self.lin_edge is not None
            # pass edge attributes to lin_edge
            # [shape: num_edges, edge_dims] --> [shape: num_edges, att_heads * in_channels]
            edge_attr = self.lin_edge(edge_attr)
            # [shape: num_edges, att_heads * in_channels] --> [num_edges, att_heads, in_channels]
            edge_attr = edge_attr.view(-1, self.heads, self.in_channels)
            # 'emulate' concatentation to node embeddings
            # [shape: num_edges, att_heads, in_channels]
            x = x + edge_attr

        # pass node embeddings through leaky relu
        x = F.leaky_relu(x, self.negative_slope)
        # multiply by node independent parameter layer. sum over in_channels.
        # x [shape: num_edges, att_heads, in_channels], self.att [shape: 1, att_heads, in_channels] --> 
        # alpha [shape: num_edges, att_heads]
        alpha = (x * self.att).sum(dim=-1)
        # calculate softmaxed attention weights
        # [shape: num_edges, att_heads]
        alpha = softmax(alpha, index, ptr, size_i)
        # store attention weights
        self._alpha = alpha
        # apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
       
        # apply attention weights to source nodes and add original source feature vector
        # [shape: num_edges, att_heads, in_channels] * [shape: num_edges, att_heads, 1] -->
        # [shape: num_edges, att_heads, in_channels]
        return torch.tile(x_j.unsqueeze(1), (1, self.heads, 1)) * (alpha.unsqueeze(-1) + 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')