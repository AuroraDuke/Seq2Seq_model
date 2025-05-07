import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import AdditiveAttention
# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1, cell_type='lstm', bidirectional=True,
                 pretrained_embedding=None, freeze_embedding=False, dropout=0.3):
        super(Encoder, self).__init__()
        self.cell_type = cell_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Print initialization info for debugging
        #print(f"Initializing Encoder with hidden_dim={hidden_dim}, bidirectional={bidirectional}")

        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)
        else:
            self.embedding = nn.Embedding(input_dim, emb_dim)

        rnn_cls = nn.LSTM if self.cell_type in ['lstm', 'bilstm'] else nn.GRU
        self.rnn = rnn_cls(
            emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional
        )

    def forward(self, x):
        # Print input shape for debugging
        #print(f"Encoder input shape: {x.shape}")
        
        embedded = self.embedding(x)  # [batch, seq_len, emb_dim]
        #print(f"Encoder embedded shape: {embedded.shape}")
        
        outputs, hidden = self.rnn(embedded)  # outputs: [batch, seq_len, hidden*dir]
        #print(f"Encoder outputs shape: {outputs.shape}")
        
        if isinstance(hidden, tuple):  # LSTM case
            h, c = hidden
            #print(f"Encoder h_n shape: {h.shape}, c_n shape: {c.shape}")
            
            # Adjust hidden state for bidirectional RNNs
            if self.bidirectional:
                # For bidirectional LSTM, we need to concatenate the last hidden state from both directions
                h_forward = h[-2]  # Last forward direction layer's hidden state
                h_backward = h[-1]  # Last backward direction layer's hidden state
                h_concat = torch.cat([h_forward, h_backward], dim=1).unsqueeze(0)
                
                c_forward = c[-2]
                c_backward = c[-1]
                c_concat = torch.cat([c_forward, c_backward], dim=1).unsqueeze(0)
                
                hidden = (h_concat, c_concat)
                #print(f"Encoder adjusted hidden shapes - h: {h_concat.shape}, c: {c_concat.shape}")
        else:  # GRU case
            #print(f"Encoder hidden shape: {hidden.shape}")
            
            if self.bidirectional:
                hidden_forward = hidden[-2]
                hidden_backward = hidden[-1]
                hidden = torch.cat([hidden_forward, hidden_backward], dim=1).unsqueeze(0)
                #print(f"Encoder adjusted hidden shape: {hidden.shape}")
                
        return outputs, hidden


# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, attention, cell_type='lstm', dropout=0.3):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.lower()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        
        # Print initialization info for debugging
        #print(f"Initializing Decoder with hidden_dim={hidden_dim}, output_dim={output_dim}")

        # Since encoder is bidirectional, encoder outputs and context will have hidden_dim*2
        encoder_features_dim = hidden_dim * 2
        
        rnn_cls = nn.LSTM if self.cell_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(encoder_features_dim, hidden_dim * 2, batch_first=True)
        
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, encoder_outputs, hidden):
        # Print input shapes for debugging
        #print(f"Decoder input - encoder_outputs: {encoder_outputs.shape}")
        #if isinstance(hidden, tuple):
            #print(f"Decoder hidden - h: {hidden[0].shape}, c: {hidden[1].shape}")
        #else:
           # print(f"Decoder hidden: {hidden.shape}")
            
        # Extract hidden state for attention
        if self.cell_type == 'lstm':
            hidden_state = hidden[0].squeeze(0)  # [batch_size, hidden_dim*2]
        else:
            hidden_state = hidden.squeeze(0)  # [batch_size, hidden_dim*2]
            
        #print(f"Decoder hidden_state for attention: {hidden_state.shape}")

        # Apply attention
        attn_weights = self.attention(hidden_state, encoder_outputs)  # [batch_size, seq_len]
        #print(f"Attention weights shape: {attn_weights.shape}")
        
        # Create context vector using attention weights
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, hidden_dim*2]
        #print(f"Context vector shape: {context.shape}")
        
        # Pass context through RNN
        output, hidden = self.rnn(context, hidden)
        #print(f"RNN output shape: {output.shape}")
        
        # Apply dropout
        output = self.dropout(output)
        
        # Make prediction
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]
        #print(f"Prediction shape: {prediction.shape}")
        
        return prediction, hidden


# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src):
        #print(f"Seq2Seq input shape: {src.shape}")
        
        # Pass through encoder
        encoder_outputs, hidden = self.encoder(src)
        #print(f"After encoder - outputs: {encoder_outputs.shape}")
        
        #if isinstance(hidden, tuple):
           # print(f"After encoder - hidden: ({hidden[0].shape}, {hidden[1].shape})")
       # else:
           # print(f"After encoder - hidden: {hidden.shape}")
        
        # Pass encoder outputs and final hidden state to decoder
        output, _ = self.decoder(encoder_outputs, hidden)
        #print(f"Final output shape: {output.shape}")
        
        return output


def build_seq2seq_model(vocab_size, output_dim, embedding_dim, hidden_dim, 
                         attention_type='additive', pretrained_embedding=None, 
                         cell_type='lstm', num_layers=1):
    """
    Build a complete Seq2Seq model with the specified attention mechanism
    
    Args:
        vocab_size: Size of the vocabulary (input dimension)
        output_dim: Number of output classes 
        embedding_dim: Dimension of the embedding layer
        hidden_dim: Dimension of the hidden layers
        attention_type: Type of attention ('additive', 'dot', or 'scaled_dot')
        pretrained_embedding: Optional pretrained embeddings
        cell_type: RNN cell type ('lstm' or 'gru')
        num_layers: Number of RNN layers
        
    Returns:
        The complete Seq2Seq model
    """
    #print(f"\nBuilding Seq2Seq model with hidden_dim={hidden_dim}, attention={attention_type}")
    
    # Create encoder
    encoder = Encoder(
        input_dim=vocab_size,
        emb_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cell_type=cell_type,
        bidirectional=True,
        pretrained_embedding=pretrained_embedding
    )
    
    # Create attention module
    if attention_type == 'additive':
        attention = AdditiveAttention(hidden_dim=hidden_dim)
    elif attention_type == 'dot':
        attention = DotProductAttention(hidden_dim=hidden_dim)
    elif attention_type == 'scaled_dot':
        attention = ScaledDotProductAttention(hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    # Create decoder
    decoder = Decoder(
        output_dim=output_dim,
        hidden_dim=hidden_dim,  # This will be doubled inside for bidirectional compatibility
        attention=attention,
        cell_type=cell_type
    )
    
    # Create Seq2Seq model
    model = Seq2Seq(encoder, decoder)
    
    return model