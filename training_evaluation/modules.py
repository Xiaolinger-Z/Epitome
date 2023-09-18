import torch
import copy
from torch.nn import functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.container import ModuleList
import torch.nn as nn
from math import sqrt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class NormGenerator(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(NormGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.gen_proj = Linear(d_model, vocab_size)

    def reset_parameters(self):
        self.gen_proj.reset_parameters()

    def forward(self, decode_output):
        """
        Generate final vocab distribution.
        :param decode_output: [T, B, H]
        :return:
        """

        gen_logits = torch.log_softmax(self.gen_proj(decode_output), dim=-1)                 # [T, B, V]
        return gen_logits


###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        if self.linear_or_not:
            stdv = 1. / sqrt(self.output_dim)
            self.linear.weight.data.uniform_(-stdv, stdv)
        else:
            stdv = 1. / sqrt(self.hidden_dim)
            for layer in range(self.num_layers - 1):
                self.linears[layer].weight.data.uniform_(-stdv, stdv)
            stdv = 1. / sqrt(self.hidden_dim)
            self.linears[-1].weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.linears[layer](h))
            h = self.linears[self.num_layers - 1](h)
            return h


class MessagePassing(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, radius):
        super(MessagePassing, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.radius = radius
        self.fc1 = nn.ModuleList([MLP(2, input_dim, output_dim, output_dim) for i in range(radius + 2)])
        self.fc2 = nn.ModuleList([MLP(2, output_dim, output_dim, output_dim) for i in range(3 * radius - 2)])
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        pass


    def forward(self, adj, features):
        l = list()
        for i in range(self.radius + 1):
            l.append(features[i])

        for i in range(2 * self.radius - 1, -1, -1):
            if i == 2 * self.radius - 1:
                if adj[i].shape != (1, 1):
                    x = self.fc1[(i + 1) // 2](l[i // 2 + 1]) + torch.spmm(adj[i],
                                                                           self.fc1[(i + 1) // 2 + 1](l[i // 2 + 1]))

                else:
                    x = self.fc1[(i + 1) // 2](l[i // 2 + 1])
            elif i % 2 == 0:
                x = self.fc1[i // 2](l[i // 2]) + torch.spmm(adj[i], self.fc2[i + i // 2](x))

            else:
                if adj[i].shape != (1, 1):
                    # adj[i] = adj[i].to_sparse_csr()
                    x = self.fc2[i + (i - 1) // 2](x) + torch.spmm(adj[i], self.fc2[i + (i - 1) // 2 + 1](x))

            x = self.dropout(x)

        return x




