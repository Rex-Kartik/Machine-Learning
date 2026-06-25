import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab, transform = None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform

        self.image_caption_pairs = []
        with open(captions_file, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                part = line.strip().split(",", 1)
                if len(part) == 2:
                    image_name, caption = part[0], part[1]
                    self.image_caption_pairs.append((image_name, caption))
        
        print(f"Loaded {len(self.image_caption_pairs)} image-caption pairs.")

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        image_name, caption = self.image_caption_pairs[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        token_ids = self.vocab.encode(caption)
        token_ids = torch.tensor(token_ids, dtype = torch.long)

        return image, token_ids
