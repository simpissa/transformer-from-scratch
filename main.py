import pickle

from src.bpe import BPE
from src.tokenizer import Tokenizer, encode_data
from src.train import train

data_path = "data/TinyStoriesV2-GPT4-valid.txt"

bpe = BPE()
vocab, merges = bpe.train_bpe(
    data_path, 
    10000, 
    ["<|endoftext|>"],
)

t = Tokenizer(vocab, merges, ["<|endoftext|>"])
with open("tokenizer/tokenizer.pkl", "wb") as f:
    pickle.dump(t, f)

encode_data("data/dataset.npy", data_path, t)

train(
        train_dataset="data/dataset.npy",
        config_path="config.yaml", 
        checkpoint_folder="checkpoints",
        use_wandb=True,
    )