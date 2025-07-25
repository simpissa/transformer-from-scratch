from typing import Iterable, Optional
import pickle
import math
import regex as re
import numpy as np

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.id_to_tok = vocab
        self.tok_to_id = {vocab[id] : id for id in vocab.keys()}
        self.merges = merges
        self.special_tokens = None
        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.split_pattern = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
        self.merges_rank = {}
        for i, merge in enumerate(self.merges):
            self.merges_rank[merge] = i


    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[list[str]] = None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pretokens = []

        if self.special_tokens is None:
            for token in re.finditer(PAT, text):
                pretokens.append([bytes([b]) for b in token.group().encode("utf-8")])
        else:
            
            parts = re.split(self.split_pattern, text)
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    for token in re.finditer(PAT, part):
                        pretokens.append([bytes([b]) for b in token.group().encode("utf-8")])
                else:
                    pretokens.append([part.encode("utf-8")])

        for i, pretoken in enumerate(pretokens):
            while True:
                max_rank = math.inf
                max_pair = None
                idx = None
                for j, pair in enumerate(zip(pretoken[:-1], pretoken[1:])):
                    rank = self.merges_rank.get(pair, math.inf)
                    if rank < max_rank:
                        max_rank = rank
                        max_pair = pair
                        idx = j
                if max_pair is None:
                    break
                pretoken[idx+1] = pretoken[idx] + pretoken[idx+1]
                pretoken.pop(idx)
            pretokens[i] = pretoken

        result = []
        # map to ids
        for pretoken in pretokens:
            for token in pretoken:
                result.append(self.tok_to_id[token])
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids: list[int]) -> str:
        result = b""
        for id in ids:
            result += self.id_to_tok[id]
        return result.decode("utf-8")

def encode_data(result_path: str, data_path: str, t: Tokenizer):
    with open(data_path) as f:
        ids = np.array(t.encode(f.read()), dtype=np.long)
    np.save(result_path, ids)

if __name__ == "__main__":
    from bpe import BPE
    bpe = BPE()
    with open("tokenizer/tokenizer.pkl", "rb") as f:
        obj = pickle.load(f)
    vocab = obj["vocab"] 
    merges = obj["merges"]
    t = Tokenizer(vocab, merges, ["<|endoftext|>"])
    with open("data/TinyStoriesV2-GPT4-valid.txt") as f:
        ids = np.array(t.encode(f.read()), dtype=np.long)
    np.save("tokenizer/tokenized_data.npy", ids)