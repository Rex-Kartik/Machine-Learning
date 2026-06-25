import json
from collections import Counter


class Vocalbulary:
    def __init__(self, freq_threshold = 5):
        self.freq_threshold = freq_threshold

        self.itos = {0: "<PAD>", 1 : "<START>", 2 : "<END>", 3 : "<UNK>"}

        self.stoi = {"<PAD>" : 0, "<START>" : 1, "<END>" : 2, "<UNK>" : 3}
    
    def build_vocabulary(self, sentences):
        word_counts = Counter()
        for sentence in sentences:
            words = sentence.lower().split()
            word_counts.update(words)
        
        for word, count in word_counts.items():
            if count >= self.freq_threshold and word not in self.stoi:
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx

        print(f"Vocabulary built: {len(self.stoi)} tokens (including specia; tokens)")

    def encode(self, sentence):
        tokens = ["<START>"]

        for word in sentence.lower().split():
            if word in self.stoi:
                tokens.append(word)
            else:
                tokens.append("<UNK>")
        
        tokens.append("<END>")
        return [self.stoi[token] for token in tokens]


    def decode(self, token_ids):
        words = []
        for idx in token_ids:
            word = self.itos[idx]
            if word == "<END>":
                break
            if word not in ["<PAD>", "<START>", "<UNK>"]:
                words.append(word)
        return " ".join(words)
    
    def __len__(self):
        return len(self.stoi)
    
    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump({"stoi" : self.stoi, "itos" : self.itos}, f)

    def load(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(filepath)

        self.itos = {int(k): v for k,v in data["itos"].items()}
        self.stoi = data["stoi"]