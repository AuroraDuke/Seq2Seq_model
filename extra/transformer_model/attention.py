import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        # Print what we're initializing with
       # print(f"Initializing AdditiveAttention with hidden_dim={hidden_dim}")
        
        # We need to handle the actual dimensions of the tensors that will be passed in
        # The hidden state will have shape [batch_size, hidden_dim*2] (if from bidirectional)
        # The encoder outputs will have shape [batch_size, seq_len, hidden_dim*2]
        self.hidden_dim = hidden_dim
        
        # Define a more flexible approach
        self.W1 = nn.Linear(hidden_dim*2, hidden_dim)  # For encoder outputs
        self.W2 = nn.Linear(hidden_dim*2, hidden_dim)  # For hidden state
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Calculates attention weights
        
        Args:
            hidden: hidden state from decoder [batch_size, hidden_dim*2]
            encoder_outputs: all hidden states from encoder [batch_size, seq_len, hidden_dim*2]
        
        Returns:
            attention weights [batch_size, seq_len]
        """
        # Print shapes for debugging
       # print(f"Attention input shapes - hidden: {hidden.shape}, encoder_outputs: {encoder_outputs.shape}")
        
        batch_size, seq_len, enc_hidden_dim = encoder_outputs.size()
        
        # Process hidden state (expand to seq_len)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim*2]
        
        # Calculate attention scores using separate transformations
        encoder_transform = self.W1(encoder_outputs)  # [batch_size, seq_len, hidden_dim]
        hidden_transform = self.W2(hidden)  # [batch_size, seq_len, hidden_dim]
        
        # Combine transformed representations
        energy = torch.tanh(encoder_transform + hidden_transform)  # [batch_size, seq_len, hidden_dim]
        
        # Calculate attention scores
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]
        
        # Normalize with softmax
        return F.softmax(attention, dim=1)


class DotProductAttention(nn.Module):
    def __init__(self, hidden_dim=None):  # hidden_dim is optional here
        super(DotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim
        # If hidden dimensions don't match between encoder and decoder
        if hidden_dim is not None:
            self.projection = nn.Linear(hidden_dim*2, hidden_dim*2)
        else:
            self.projection = None
        
    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: decoder hidden state [batch_size, hidden_dim*2]
            encoder_outputs: encoder outputs [batch_size, seq_len, hidden_dim*2]
        """
        # Project hidden if needed
        if self.projection is not None:
            hidden = self.projection(hidden)
        
        # Reshape hidden for batched matrix multiplication
        hidden = hidden.unsqueeze(2)  # [batch_size, hidden_dim*2, 1]
        
        # Calculate attention scores
        attention = torch.bmm(encoder_outputs, hidden).squeeze(2)  # [batch_size, seq_len]
        
        # Normalize with softmax
        return F.softmax(attention, dim=1)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim=None):  # hidden_dim is optional here
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim
        # If hidden dimensions don't match
        if hidden_dim is not None:
            self.projection = nn.Linear(hidden_dim*2, hidden_dim*2)
        else:
            self.projection = None
        
    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: decoder hidden state [batch_size, hidden_dim*2]
            encoder_outputs: encoder outputs [batch_size, seq_len, hidden_dim*2]
        """
        # Project hidden if needed
        if self.projection is not None:
            hidden = self.projection(hidden)
        
        # Reshape hidden
        hidden = hidden.unsqueeze(2)  # [batch_size, hidden_dim*2, 1]
        
        # Calculate scaled attention scores
        scale = torch.sqrt(torch.tensor(float(encoder_outputs.size(-1))))
        attention = torch.bmm(encoder_outputs, hidden).squeeze(2) / scale
        
        # Normalize with softmax
        return F.softmax(attention, dim=1)