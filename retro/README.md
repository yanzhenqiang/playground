## Transformer and MMCTS
q a ————> a^ , new data
https://github.com/ZiwenZhuang/MMCTS/blob/master/memory.py
https://github.com/EndingCredits/Neural-Episodic-Control
https://github.com/YeWR/EfficientZero
https://github.com/lucidrains/memorizing-transformers-pytorch/blob/main/train.py

https://arxiv.org/abs/2112.04426
.,
Deepmind's Retrieval based Attention net, in Pytorch. This will deviate from the paper slightly, using rotary embeddings for relative positional encoding, as well as Faiss library instead of Scann.

https://github.com/criteo/autofaiss

for building the index and calculating the k-nearest neighbors for all chunks.

http://jalammar.github.io/illustrated-retrieval-transformer/

https://arxiv.org/abs/2009.06857 definitely deserved

https://arxiv.org/abs/2203.00555 DeepNet paper

## Usage

```python
import torch
import RETRO

retro = RETRO(
    chunk_size = 64,                         # the chunk size that is indexed and retrieved
    max_seq_len = 2048,                      # max sequence length
    enc_dim = 896,                           # encoder model dim
    enc_depth = 2,                           # encoder depth
    dec_dim = 796,                           # decoder model dim
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (3, 6, 9, 12),   # decoder cross attention layers (with causal chunk cross attention)
    heads = 8,                               # attention heads
    dim_head = 64,                           # dimension per head
    dec_attn_dropout = 0.25,                 # decoder attention dropout
    dec_ff_dropout = 0.25,                   # decoder feedforward dropout
    use_deepnet = True                       # turn on post-normalization with DeepNet
)

seq = torch.randint(0, 20000, (2, 2048 + 1))      # plus one since it is split into input and labels for training
retrieved = torch.randint(0, 20000, (2, 32, 2, 128)) # retrieved tokens - (batch, num chunks, num retrieved neighbors, retrieved chunk with continuation)

loss = retro(seq, retrieved, return_loss = True)
loss.backward()
```


## RETRO Datasets

The `RETRODataset` class accepts paths to a number of memmapped numpy arrays containing the chunks, the index of the first chunk in the sequence to be trained on (in RETRO decoder), and the pre-calculated indices of the k-nearest neighbors per chunk.

You can use this to easily assemble the data for `RETRO` training, if you do not wish to use the `TrainingWrapper` from above.

Furthermore, all the functions needed to create the necessary memmapped data is in the sections to follow.


```python
import torch
from torch.utils.data import DataLoader
from retro_pytorch import RETRO, RETRODataset

# mock data constants

import numpy as np

NUM_CHUNKS = 1000
CHUNK_SIZE = 64
NUM_SEQS = 100
NUM_NEIGHBORS = 2

def save_memmap(path, tensor):
    f = np.memmap(path, dtype = tensor.dtype, mode = 'w+', shape = tensor.shape)
    f[:] = tensor
    del f

# generate mock chunk data

save_memmap(
    './train.chunks.dat',
    np.int32(np.random.randint(0, 8192, size = (NUM_CHUNKS, CHUNK_SIZE + 1)))
)

# generate nearest neighbors for each chunk

save_memmap(
    './train.chunks.knn.dat',
    np.int32(np.random.randint(0, 1000, size = (NUM_CHUNKS, NUM_NEIGHBORS)))
)

# generate seq data

save_memmap(
    './train.seq.dat',
    np.int32(np.random.randint(0, 128, size = (NUM_SEQS,)))
)

# instantiate dataset class
# which constructs the sequence and neighbors from memmapped chunk and neighbor information

train_ds = RETRODataset(
    num_sequences = NUM_SEQS,
    num_chunks = NUM_CHUNKS,
    num_neighbors = NUM_NEIGHBORS,
    chunk_size = CHUNK_SIZE,
    seq_len = 2048,
    chunk_memmap_path = './train.chunks.dat',
    chunk_nn_memmap_path = './train.chunks.knn.dat',
    seq_memmap_path = './train.seq.dat'
)

train_dl = iter(DataLoader(train_ds, batch_size = 2))

# one forwards and backwards

retro = RETRO(
    max_seq_len = 2048,                      # max sequence length
    enc_dim = 896,                           # encoder model dimension
    enc_depth = 3,                           # encoder depth
    dec_dim = 768,                           # decoder model dimensions
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
    heads = 8,                               # attention heads
    dim_head = 64,                           # dimension per head
    dec_attn_dropout = 0.25,                 # decoder attention dropout
    dec_ff_dropout = 0.25                    # decoder feedforward dropout
).cuda()

seq, retrieved = map(lambda t: t.cuda(), next(train_dl))

# seq       - (2, 2049)         - 1 extra token since split by seq[:, :-1], seq[:, 1:]
# retrieved - (2, 32, 2, 128)   - 128 since chunk + continuation, each 64 tokens

loss = retro(
    seq,
    retrieved,
    return_loss = True
)

loss.backward()

```

## Retrieval related tools

This repository will use the default tokenizer (sentencepiece) for the cased version of BERT. Embeddings will be fetched from the vanilla BERT, and can either be masked mean pooled representation, or the CLS token.

ex. masked mean pooled representation

```python
from retro_pytorch.retrieval import bert_embed, tokenize

ids = tokenize([
    'hello world',
    'foo bar'
])

embeds = bert_embed(ids) # (2, 768) - 768 is hidden dimension of BERT
```

ex. CLS token representation


```python
from retro_pytorch.retrieval import bert_embed, tokenize

ids = tokenize([
    'hello world',
    'foo bar'
])

embeds = bert_embed(ids, return_cls_repr = True) # (2, 768)
```

Create your chunks and chunk start indices (for calculating sequence ranges for autoregressive training) using `text_folder_to_chunks_`

```python
from retro_pytorch.retrieval import text_folder_to_chunks_

stats = text_folder_to_chunks_(
    folder = './text_folder',
    glob = '**/*.txt',
    chunks_memmap_path = './train.chunks.dat',
    seqs_memmap_path = './train.seq.dat',
    doc_ids_memmap_path = './train.doc_ids.dat',  # document ids are needed for filtering out neighbors belonging to same document appropriately during computation of nearest neighbors
    chunk_size = 64,
    seq_len = 2048,
    max_chunks = 1_000_000,
    max_seqs = 100_000
)

# {'chunks': <number of chunks>, 'docs': <number of documents>, 'seqs': <number of sequences>}
```

## Fetching Nearest Neighbors

You can turn your memmapped chunks numpy array into embeddings and a faiss index with one command

```python
from retro_pytorch.retrieval import chunks_to_index_and_embed

index, embeddings = chunks_to_index_and_embed(
    num_chunks = 1000,
    chunk_size = 64,
    chunk_memmap_path = './train.chunks.dat'
)

query_vector = embeddings[:1]                   # use first embedding as query
_, indices = index.search(query_vector, k = 2)  # fetch 2 neighbors, first indices should be self

neighbor_embeddings = embeddings[indices]       # (1, 2, 768)

```

You can also directly calculate the nearest neighbor file necessary for training, with `chunks_to_precalculated_knn_` command

```python
from retro_pytorch.retrieval import chunks_to_precalculated_knn_

chunks_to_precalculated_knn_(
    num_chunks = 1000,
    chunk_size = 64,
    chunk_memmap_path = './train.chunks.dat',    # path to main chunks dataset
    doc_ids_memmap_path = './train.doc_ids.dat', # path to document ids created by text_folder_to_chunks_, used for filtering out neighbors that belong to the same document
    num_nearest_neighbors = 2,                   # number of nearest neighbors you'd like to use
    num_extra_neighbors = 10                     # fetch 10 extra neighbors, in the case that fetched neighbors are frequently from same document (filtered out)
)

# nearest neighbor info saved to ./train.chunks.knn.dat

```

## Citations

```bibtex
@misc{borgeaud2022improving,
    title   = {Improving language models by retrieving from trillions of tokens}, 
    author  = {Sebastian Borgeaud and Arthur Mensch and Jordan Hoffmann and Trevor Cai and Eliza Rutherford and Katie Millican and George van den Driessche and Jean-Baptiste Lespiau and Bogdan Damoc and Aidan Clark and Diego de Las Casas and Aurelia Guy and Jacob Menick and Roman Ring and Tom Hennigan and Saffron Huang and Loren Maggiore and Chris Jones and Albin Cassirer and Andy Brock and Michela Paganini and Geoffrey Irving and Oriol Vinyals and Simon Osindero and Karen Simonyan and Jack W. Rae and Erich Elsen and Laurent Sifre},
    year  = {2022},
    eprint = {2112.04426},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@misc{su2021roformer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
    year    = {2021},
    eprint  = {2104.09864},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@article{Wang2022DeepNetST,
    title   = {DeepNet: Scaling Transformers to 1, 000 Layers},
    author  = {Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Dongdong Zhang and Furu Wei},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2203.00555}
}
```

```bibtex
@misc{zhang2021sparse,
    title   = {Sparse Attention with Linear Units},
    author  = {Biao Zhang and Ivan Titov and Rico Sennrich},
    year    = {2021},
    eprint  = {2104.07012},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```