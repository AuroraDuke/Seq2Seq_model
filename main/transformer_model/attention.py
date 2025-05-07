
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention, dim=1)


class DotProductAttention(nn.Module):
    def forward(self, hidden, encoder_outputs):
        if hidden.dim() == 2:  # shape: (batch_size, hidden_dim)
            hidden = hidden.unsqueeze(2)  # shape: (batch_size, hidden_dim, 1)
        attention = torch.bmm(encoder_outputs, hidden).squeeze(2)  # (batch_size, seq_len)
        return F.softmax(attention, dim=1)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))

    def forward(self, hidden, encoder_outputs):
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(2)
        attn_scores = torch.bmm(encoder_outputs, hidden).squeeze(2)
        return F.softmax(attn_scores / self.scale.to(hidden.device), dim=1)
