import os
from typing import Tuple, BinaryIO, Any
from multiprocessing import cpu_count, Pool
from functools import reduce, partial
import heapq
import pickle

import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class ReverseOrder:
    def __init__(self, pair):
        self.pair = pair
    def __lt__(self, other):
        return self.pair > other.pair
    def __eq__(self, other):
        return self.pair == other.pair

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def append(self, pair: Tuple[bytes]):
        node = Pair(pair, self)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
        return node

class Pair:
    def __init__(self, pair: Tuple[bytes], owner: LinkedList):
        self.pair = pair
        self.next = None
        self.prev = None
        self.owner = owner
    def __str__(self):
        return self.pair

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize_chunk(start: int, end: int, input_path: str, split_special_tokens: list[str]):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        pretokens = {}
        split_pattern = "|".join(map(re.escape, split_special_tokens))
        parts = re.split(split_pattern, chunk)
        for part in parts:
            for token in re.finditer(PAT, part):
                key = tuple(bytes([b]) for b in token.group().encode("utf-8"))
                pretokens[key] = pretokens.get(key, 0) + 1

        return pretokens

def merge_dicts(d1: dict, d2: dict, default: Any):
    result = d1.copy()
    for key in d2.keys():
        result[key] = d2.get(key, default)
    return result

class BPE:
    def __init__(self):
        self.linked_lists = {}
        self.pairs = {}
        self.pair_idxs = {}
        self.pair_heap = []

    def create_pairs(self, pretokens: dict[str, int]):
        for pretoken, freq in pretokens.items():
            l = LinkedList()
            for b1, b2 in zip(pretoken[:-1], pretoken[1:]):
                pair = (b1, b2)
                self.pairs[pair] = self.pairs.get(pair, 0) + freq
                node = l.append(pair)
                if pair not in self.pair_idxs:
                    self.pair_idxs[pair] = []
                self.pair_idxs[pair].append(node)
            self.linked_lists[l] = freq

    def create_heap(self):
        for pair, freq in self.pairs.items():
            heapq.heappush(self.pair_heap, (-freq, ReverseOrder(pair)))

    def find_max_pair(self):
        while self.pair_heap:
            nfreq, rpair = heapq.heappop(self.pair_heap)
            freq = -nfreq
            pair = rpair.pair
            if self.pairs.get(pair, 0) == freq:
                return pair

    def _merge(self, old: Pair, new: Tuple[bytes], freq: int):
        # update pair freq
        self.pairs[new] = self.pairs.get(new, 0) + freq
        heapq.heappush(self.pair_heap, (-self.pairs[new], ReverseOrder(new)))
        self.pairs[old.pair] -= freq
        if self.pairs[old.pair] == 0:
            del self.pairs[old.pair]
        else:
            heapq.heappush(self.pair_heap, (-self.pairs[old.pair], ReverseOrder(old.pair)))
        
        # update idx
        self.pair_idxs[old.pair].remove(old)
        if not self.pair_idxs[old.pair]:
            del self.pair_idxs[old.pair]

        if new not in self.pair_idxs:
            self.pair_idxs[new] = []
        self.pair_idxs[new].append(old)
            

    def merge(self, max_pair: Tuple[bytes]):
        new_token = max_pair[0] + max_pair[1]
        for merged_pair in self.pair_idxs[max_pair]:
            # a,b,c,d -> a,bc,d
            freq = self.linked_lists[merged_pair.owner]
            prev = merged_pair.prev
            next = merged_pair.next
            if prev is not None:
                # a,bc
                new_pair_left = (prev.pair[0], new_token)
                self._merge(prev, new_pair_left, freq)
                # update linked_list
                prev.pair = new_pair_left
                prev.next = next

            if next is not None:
                # bc,d
                new_pair_right = (new_token, next.pair[1])
                self._merge(next, new_pair_right, freq)
                # update linked_list
                next.pair = new_pair_right
                next.prev = prev
        self.pairs.pop(max_pair)
        self.pair_idxs.pop(max_pair)

    def train_bpe(
        self,
        input_path: str, 
        vocab_size: int, 
        special_tokens: list[str],
    ) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        
        num_procs = cpu_count()
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_procs, b"<|endoftext|>")
        # pretokenize
        args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            args.append((start, end, input_path, special_tokens))
        with Pool(processes=num_procs) as pool:
            results = pool.starmap(pretokenize_chunk, args)
        pretokens = reduce(partial(merge_dicts, default=0), results, {})
        self.create_pairs(pretokens)
        self.create_heap()

        vocab = {}
        # initialize vocab
        idx = 0
        for special_token in special_tokens:
            vocab[idx] = special_token.encode("utf-8")
            idx += 1
        for i in range(256):
            vocab[idx] = bytes([i])
            idx += 1

        # train
        merges = []
        while len(vocab) < vocab_size:
            max_pair = self.find_max_pair()
            self.merge(max_pair)
            merges.append(max_pair)
            vocab[idx] = max_pair[0] + max_pair[1]
            idx += 1

        return vocab, merges


if __name__ == "__main__":
    bpe = BPE()
    vocab, merges = bpe.train_bpe(
        "data/TinyStoriesV2-GPT4-valid.txt", 
        10000, 
        ["<|endoftext|>"],
    )
    tokenizer = {"vocab": vocab, "merges": merges}
    with open("tokenizer/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)