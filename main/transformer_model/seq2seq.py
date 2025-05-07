import torch
import torch.nn as nn
from .attention import AdditiveAttention, DotProductAttention, ScaledDotProductAttention
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, cell_type='lstm', bidirectional=False, pretrained_embedding=None, freeze_embedding=False, attention=None):
        super(Encoder, self).__init__()
        self.cell_type = cell_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.attention = attention

        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)
        else:
            self.embedding = nn.Embedding(input_dim, emb_dim)

        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
        elif self.cell_type == 'bilstm':
            self.rnn = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type")

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, attention, cell_type='lstm'):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.lower()
        self.attention = attention

        self.rnn_input_dim = hidden_dim  # due to concatenation
        self.embedding = nn.Identity()  # not used for classification

        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.rnn_input_dim, hidden_dim, batch_first=True)
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(self.rnn_input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type")

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_outputs, hidden):
        if self.cell_type == 'lstm':
            hidden_state = hidden[0][-1]  # last layer hidden state
        else:
            hidden_state = hidden[-1]     # for GRU

        attn_weights = self.attention(hidden_state, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden_dim)
        rnn_input = context  # already (batch, 1, hidden_dim)

        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src):
        encoder_outputs, hidden = self.encoder(src)
        output, _ = self.decoder(encoder_outputs, hidden)
        return output