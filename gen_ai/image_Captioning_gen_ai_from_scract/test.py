from models.decoder import Decoder
import torch

decoder = Decoder(embed_size= 256, hidden_size= 512, vocal_size= 10000)
dummy_feat = torch.randn(4, 2048)
dummy_cap = torch.randint(0, 10000, (4,15))
out = decoder(dummy_feat, dummy_cap)
print(out.shape )