# Getting Started

## Installing Faiss

```shell
# 新建一个干净的环境
conda create -n faiss_env python=3.9 -y
# 切换
conda activate faiss_env
```

根据需要安装CPU或GPU版本的Faiss

```shell
# CPU-only version
conda install -c pytorch faiss-cpu=1.10.0

# GPU(+CPU) version
conda install -c pytorch -c nvidia faiss-gpu=1.10.0

# GPU(+CPU) version with NVIDIA cuVS
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.10.0

# GPU(+CPU) version using AMD ROCm not yet available
```

> [更多关于安装的信息](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

## Basic Usage

```python
import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

### Getting some data

```python
import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
```

- `d`：向量的维度
- `nb`：数据库的大小
- `nq`：查询的数量
- `xb`：数据库向量
- `xq`：查询向量

`xb[:, 0] += np.arange(nb) / 1000.`：构造具有一定分布规律的数据，使与i号查询相近的向量就在数据库中i号附近。

### Building an index and adding the vectors to it

```python
import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)
```

Faiss 是围绕 Index 对象构建的。它封装数据库向量集，并选择性地对它们进行预处理以提高搜索效率。

所有索引在构建时都需要知道它们操作的向量的维度。大多数索引还需要一个训练阶段，以分析向量的分布。对于 `IndexFlatL2`可以跳过这个操作。

当索引构建并训练完成后，可以对索引执行两个操作：add 和 search 。

要向索引中添加元素，调用 `add` 并传入 `xb`。还可以查看索引的两个状态变量：`is_trained`（指示是否需要训练）和 `ntotal`（索引中存储的向量数量）。

### Searching

```python
k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```



## Faster search

## Lower memory footprint