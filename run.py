import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.register_buffer('PE', self._create_positional_encoding())
    
    def _create_positional_encoding(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
    
    def forward(self, x):
        return x + self.PE[:x.size(1), :]

def scaled_dot_product(q, k, v, mask=None):
    # q k, v = 30 * 8 * 200  * 64
    d_k = q.size()[-1] # attention head whc in ds case is 64
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k) # 30 * 8 * 200 * 200
    if mask is not None:
        scaled += mask # 30 * 8 * 200 * 200 
    attention = F.softmax(scaled, dim=-1) # 30 * 8 * 200 * 200
    values = torch.matmul(attention, v) # 30 * 8 * 200 * 64
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model # 512    
        self.num_heads = num_heads # 8
        self.head_dim = d_model // num_heads # 64
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model) # 512 * 1536
        self.linear_layer = nn.Linear(d_model, d_model) # 512 * 512
        # Add positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_sequence_length=5000)
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size() # 30 * 200 * 512  
        
        # Apply positional encoding first
        if x.size(-1) == self.d_model:  # Only apply if dimensions match
            x = self.pos_encoding(x)
            
        qkv = self.qkv_layer(x) # 30 * 200 * 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 30 * 200 * 8 *192
        qkv = qkv.permute(0, 2, 1, 3) # 30 * 8 * 200 * 192
        q, k, v = qkv.chunk(3, dim=-1) # 30 * 8 * 200 * 64
        values, attention = scaled_dot_product(q, k, v, mask) ## value 30 * 8 * 200 * 64  ## attention 30 * 8 * 200 * 200
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim) # 30 * 200 * 512
        out = self.linear_layer(values) # 30 * 200 * 512
        return out, attention


# class LayerNormalization():
#     def __init__(self, parameters_shape, eps=1e-5):
#         self.parameters_shape=parameters_shape
#         self.eps=eps
#         self.gamma = nn.Parameter(torch.ones(parameters_shape))
#         self.beta =  nn.Parameter(torch.zeros(parameters_shape))

#     def forward(self, inputs):
#         dims = [-(i + 1) for i in range(len(self.parameters_shape))]
#         mean = inputs.mean(dim=dims, keepdim=True)
#         print(f"Mean \n ({mean.size()}): \n {mean}")
#         var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
#         std = (var + self.eps).sqrt()
#         print(f"Standard Deviation \n ({std.size()}): \n {std}")
#         y = (inputs - mean) / std
#         print(f"y \n ({y.size()}) = \n {y}")
#         out = self.gamma * y  + self.beta
#         print(f"out \n ({out.size()}) = \n {out}")
#         # Normalize attention scores if provided
#         if attention is not None:
#             # Normalize attention scores along the sequence length dimension
#             attention_mean = attention.mean(dim=-1, keepdim=True)
#             attention_var = ((attention - attention_mean) ** 2).mean(dim=-1, keepdim=True)
#             attention_std = (attention_var + self.eps).sqrt()
#             normalized_attention = (attention - attention_mean) / attention_std
#             return out, normalized_attention
            
#         return out

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape # 512    
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # 512
        self.beta =  nn.Parameter(torch.zeros(parameters_shape)) # 512

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))] #[-1]
        mean = inputs.mean(dim=dims, keepdim=True) # 30 * 200 * 1   
        print(f"Mean ({mean.size()})")
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True) # 30 * 200 * 1
        std = (var + self.eps).sqrt() # 30 * 200 * 1
        print(f"Standard Deviation  ({std.size()})")
        y = (inputs - mean) / std # 30 * 200 * 512
        print(f"y: {y.size()}")
        out = self.gamma * y  + self.beta # 30 * 200 * 512
        print(f"self.gamma: {self.gamma.size()}, self.beta: {self.beta.size()}")
        print(f"out: {out.size()}")
        return out


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # 512 * 2048  
        self.linear2 = nn.Linear(hidden, d_model) # 2048 * 512  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x): # 30 * 200 * 512  
        x = self.linear1(x) # 30 * 200 * 2048      
        print(f"x after first linear layer: {x.size()}")
        x = self.relu(x) # 30 * 200 * 2048
        print(f"x after activation: {x.size()}")
        x = self.dropout(x) # 30 * 200 * 2048
        print(f"x after dropout: {x.size()}")
        x = self.linear2(x) # 30 * 200 * 512
        print(f"x after 2nd linear layer: {x.size()}")
        return x
    

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(input_dim=d_model, d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x # 30 * 200 * 512 
        print("------- ATTENTION 1 ------")
        # x = self.attention(x, mask=None) # 30 * 200 * 512
        attention_output, attention_scores = self.attention(x, mask=None)
        print("------- DROPOUT 1 ------")
        # x = self.dropout1(x) # 30 * 200 * 512
        attention_output = self.dropout1(attention_output)
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        # x = self.norm1(x + residual_x) # 30 * 200 * 512
        x = self.norm1(attention_output + residual_x)
        residual_x = x # 30 * 200 * 512 
        print("------- ATTENTION 2 ------")
        x = self.ffn(x) # 30 * 200 * 512
        print("------- DROPOUT 2 ------")
        x = self.dropout2(x) # 30 * 200 * 512
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        x = self.norm2(x + residual_x) # 30 * 200 * 512
        return x
    

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                     for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x


# Example usage
input_dim = 1024
d_model = 512
num_heads = 8
batch_size = 30
sequence_length = 5
x = torch.randn((batch_size, sequence_length, input_dim))
model = MultiheadAttention(input_dim, d_model, num_heads)
out = model.forward(x)




# ... existing code ...

if __name__ == "__main__":
    # Test parameters
    input_dim = 512  # Changed to match d_model for simplicity
    d_model = 512
    ffn_hidden = 2048
    num_heads = 8
    batch_size = 2
    sequence_length = 4
    drop_prob = 0.1
    num_layers = 2

    # Create test input
    x = torch.randn((batch_size, sequence_length, input_dim))
    print("\n=== Input ===")
    print(f"Input shape: {x.shape}")
    
    # Test Positional Encoding
    print("\n=== Testing Positional Encoding ===")
    pos_enc = PositionalEncoding(d_model, max_sequence_length=100)
    pos_output = pos_enc(x)
    print(f"Positional Encoding output shape: {pos_output.shape}")
    
    # Test Multi-head Attention
    print("\n=== Testing Multi-head Attention ===")
    mha = MultiheadAttention(input_dim, d_model, num_heads)
    mha_output, attention = mha(x)
    print(f"Multi-head Attention output shape: {mha_output.shape}")
    print(f"Attention shape: {attention.shape}")
    
    # Test Layer Normalization
    print("\n=== Testing Layer Normalization ===")
    norm = LayerNormalization(parameters_shape=[d_model])
    norm_output = norm(mha_output)
    print(f"Layer Normalization output shape: {norm_output.shape}")
    print(f"Layer Normalization stats - Mean: {norm_output.mean():.4f}, Std: {norm_output.std():.4f}")
    
    # Test Position-wise Feed Forward
    print("\n=== Testing Position-wise Feed Forward ===")
    ff = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
    ff_output = ff(norm_output)
    print(f"Feed Forward output shape: {ff_output.shape}")
    
    # Test Encoder Layer
    print("\n=== Testing Encoder Layer ===")
    encoder_layer = EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
    encoder_layer_output = encoder_layer(x)
    print(f"Encoder Layer output shape: {encoder_layer_output.shape}")
    
    # Test Full Encoder
    print("\n=== Testing Full Encoder ===")
    encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
    encoder_output = encoder(x)
    print(f"Full Encoder output shape: {encoder_output.shape}")
    
    # Verify shapes are consistent
    print("\n=== Shape Consistency Check ===")
    print(f"Input shape: {x.shape}")
    print(f"Final output shape: {encoder_output.shape}")
    assert x.shape == encoder_output.shape, "Input and output shapes should match!"
    print("All shape checks passed!")




# # Test code
# if __name__ == "__main__":
#     # Create test input
#     input_dim = 512
#     d_model = 512
#     num_heads = 8
#     batch_size = 30
#     sequence_length = 200
    
#     print("\n=== Testing LayerNormalization ===")
#     norm_layer = LayerNormalization(parameters_shape=[d_model])

#     # Create random input
#     x = torch.randn((batch_size, sequence_length, input_dim))
#     print(f"\nInput shape: {x.shape}")
#     print(f"Sample input values:\n{x[0, 0, :10]}")


    
#     # Create and run model
#     model = MultiheadAttention(input_dim, d_model, num_heads)
#     out, attention = model.forward(x)

#      # Apply normalization
#     normalized_out = norm_layer(out)

#     print(f"\nNormalized output shape: {normalized_out.shape}")
#     print(f"Normalized output mean: {normalized_out.mean():.4f}")
#     print(f"Normalized output std: {normalized_out.std():.4f}")


#      # Test Positional Encoding
#     print("\n=== Testing Positional Encoding ===")
#     pos_enc = PositionalEncoding(d_model, max_sequence_length=100)
#     pos_output = pos_enc(x)
#     print(f"Positional Encoding output shape: {pos_output.shape}")
    
#     # Print output information
#     print(f"\nOutput shape: {out.shape}")
#     print(f"Sample output values:\n{out[0, 0, :10]}")
#     print(f"\nAttention shape: {attention.shape}")
#     print(f"\nOutput mean: {out.mean():.4f}")
#     print(f"Output std: {out.std():.4f}")
