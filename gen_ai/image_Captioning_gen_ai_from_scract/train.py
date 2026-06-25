import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.encoder import TinyCNN
from models.decoder import Decoder
from utils.vocabulary import Vocalbulary
from utils.data_loader import get_data_loader
from utils.helpers import collate_fn

def main():
    captions_file = "data/flickr8k/captions.txt"
    image_dir = "data/flickr8k/Images"

    print("Building vocabulary....")
    with open(captions_file, 'r') as f:
        lines = f.readlines()[1:]
        all_captions = [line.strip().split(",", 1)[1] for line in lines]
    
    vocab = Vocalbulary(freq_threshold = 5)
    vocab.build_vocabulary(all_captions)
    print(f"Vocabulary size: {len(vocab)}")

    print("Loading data...")
    batch_size = 64
    dataLoader = get_data_loader(
        image_dir=image_dir,
        captions_file=captions_file,
        vocab=vocab,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    print(f"DataLoader has {len(dataLoader)} batches.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading encoder...")
    encoder = TinyCNN().to(device)

    cnn_weights_path = 'cnn_model.pth'
    state_dict = torch.load(cnn_weights_path,  map_location=device)
    encoder.load_state_dict(state_dict, strict= False)
    print("Encoder weights loaded successfully.")
    
    print("Initializing decoder...")
    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)

    decoder = Decoder(
        embed_size= embed_size,
        hidden_size= hidden_size,
        vocal_size=vocab_size,
        feature_size=2048
    ).to(device)
    decoder_weights_path = "decoder_captioning.pth"
    try:
        decoder.load_state_dict(torch.load(decoder_weights_path, map_location=device))
    except Exception as e:
        print("No weigths found for decoder training from the start")

    print(f"Decoder has {sum(p.numel() for p in decoder.parameters()):,} parameters")


    criterion = nn.CrossEntropyLoss(ignore_index = vocab.stoi["<PAD>"])
    optimezer = optim.AdamW(decoder.parameters(), lr=0.001)

    num_epochs = 10
    print(f"Starting Training...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/ {num_epochs}")
        encoder.eval()
        decoder.train()
        running_loss = 0.0

        for i, (images, captions, lengths) in enumerate(dataLoader):
            images = images.to(device)
            captions = captions.to(device)

            with torch.no_grad():
                features = encoder.forward_features(images)
            
            outputs = decoder(features, captions)

            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))

            optimezer.zero_grad()
            loss.backward()
            optimezer.step()

            running_loss += loss.item()

            if(i + 1) % 100 == 0:
                print(f" Batch {i + 1} / {len(dataLoader)}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataLoader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    torch.save(decoder.state_dict(), 'decoder_captioning.pth')
    print(f"Decoder weights saved as 'decoder_captioning.pth")

    print("Training Complete!")



if __name__ == "__main__":
    main()