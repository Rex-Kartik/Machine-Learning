import torch
from PIL import Image
import torchvision.transforms as transforms

from models.encoder import TinyCNN
from models.decoder import Decoder
from utils.vocabulary import Vocalbulary

def generate_caption(image_path, encoder, decoder, vocab, device, max_len=20):
    tranform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = tranform(image).unsqueeze(0).to(device)

    with torch.no_grad():
         features = encoder.forward_features(image_tensor)
    
    h_prev = decoder.projection(features)
    c_prev = torch.zeros_like(h_prev)
    current_token = torch.tensor([vocab.stoi["<START>"]], dtype= torch.long).to(device)
    generated_tokens = [current_token.item()]
    with torch.no_grad():
        for i in range(max_len):
            embedded_vector = decoder.embed(current_token)
            h_prev, c_prev = decoder.lstm_cell(embedded_vector, h_prev, c_prev)
            logits = decoder.fc(h_prev)
            word = torch.argmax(logits)

            if(word.item() == vocab.stoi["<END>"]):
                break

            generated_tokens.append(word.item())
            current_token = word.unsqueeze(0)
        
    return vocab.decode(generated_tokens)



def main():
    print("1. Setting up device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using {device}")

    print("Building vocabulary....")
    with open("data/flickr8k/captions.txt", "r") as f:
        lines = f.readlines()[1:]
        all_captions = [line.strip().split(",", 1)[1] for line in lines]
    vocab = Vocalbulary(freq_threshold= 5)
    vocab.build_vocabulary(all_captions)
    print(f"Vocabulary size: {len(vocab)}")

    print(f"Loading encoder...")
    encoder = TinyCNN().to(device)

    CNN_weights_path = "cnn_model.pth"
    encoder.load_state_dict(torch.load(CNN_weights_path, map_location=device, weights_only=True), strict = False)
    print("Encoder weights loaded successfully.")

    print("Initializing decoder...")
    decoder = Decoder(
        embed_size= 256,
        hidden_size= 512,
        vocal_size= len(vocab),
        feature_size=2048
    ).to(device)
    decoder.load_state_dict(torch.load("decoder_captioning.pth", map_location=device, weights_only=True))
    print("Decoder weights loaded successfully.")

    image_path = input()
    caption = generate_caption(
        image_path,
        encoder,
        decoder,
        vocab,
        device
    )

    Image.open(image_path).show()
    print(f"\n GEnrated Caption: {caption}")

if __name__ == "__main__":
    main()