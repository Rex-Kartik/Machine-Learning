import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.dataset import Flickr8kDataset
from utils.helpers import collate_fn


def get_data_loader(image_dir, captions_file, vocab, batch_size = 64, shuffle = True, num_workers = 0):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])

    dataset = Flickr8kDataset(image_dir, captions_file, vocab, transform= transform)

    dataloader = DataLoader(
        dataset= dataset,
        batch_size= batch_size,
        shuffle= shuffle,
        num_workers= num_workers,
        collate_fn= collate_fn,
        pin_memory= True
    )

    return dataloader