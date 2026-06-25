import torch

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]

    lengths = [len(seq) for seq in captions]
    max_len = max(lengths)


    padded_captions = torch.zeros(len(captions), max_len, dtype = torch.long)
    for i, seq in enumerate(captions):
        padded_captions[i, :len(seq)] = seq

    return images, padded_captions, lengths