import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import MessagePassing

class k_hop_GraphNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, dropout, radius):
        super(k_hop_GraphNN, self).__init__()

        self.dec_voc = vocab_size
        self.mp1 = MessagePassing(input_dim, hidden_dim, dropout, radius)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def reset_parameters(self):

        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.fc1.reset_parameters()


    def forward(self, adj, final_features, segment, idx):


        x = self.mp1(adj, final_features)

        x = self.bn1(x)
        idx = torch.transpose(idx.repeat(x.size()[1],1), 0, 1)
        out = torch.zeros(torch.max(idx)+1, x.size()[1]).cuda()
        out = out.scatter_add_(0, idx, x)
        out = self.bn2(out)
        out = self.relu(self.fc1(out))
        out2 = self.dropout(out)

        return out2

