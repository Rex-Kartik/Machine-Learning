from utils.vocabulary import Vocalbulary
from utils.data_loader import get_data_loader
import torch

captions_file = 'data/flickr8k/captions.txt'

with open(captions_file, 'r') as f:
    lines = f.readlines()[1:]
    all_captions = [line.strip().split(",", 1)[1] for line in lines]

vocab = Vocalbulary(freq_threshold=5)
vocab.build_vocabulary(all_captions)
print(f"Vocabilary size: {len(vocab)}")

image_dir = "data/flickr8k/Images"
batch_size = 4

dataloader = get_data_loader(
    image_dir= image_dir,
    captions_file=captions_file,
    vocab= vocab,
    batch_size= batch_size,
    shuffle= True
)


image, padded_captions, lengths = next(iter(dataloader))

print(f"Image shape: {image.shape}")
print(f"Padded caption shape: {padded_captions}")
print(f"Caption lengths: {lengths}")


first_captions_tokens = padded_captions[0][:lengths[0]]
decoded = vocab.decode(first_captions_tokens.tolist())
print(f"\n Decoded caption 1: {decoded}")