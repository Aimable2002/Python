
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
