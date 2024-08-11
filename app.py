# import sllm2
import torch
import torch.nn as nn
import sllm2

with open("./data_cache.txt") as f:
    db = f.read()

vocab = sorted(set(db))
vocab_size = len(vocab)

itos = { i:j for i,j in enumerate(vocab)}
stoi = { j:i for i,j in enumerate(vocab)}

encode = lambda x: [itos[i] for i in x]
decode = lambda x: [stoi[i] for i in x]

print( f"{vocab_size = }" , vocab)
# print(e:=encode([1,2,3,4,5,6,7,6]))
# print(decode(e))


