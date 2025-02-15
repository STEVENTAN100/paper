# Getting Started

全文字数: **{{ #word_count }}**

阅读时间: **{{ #reading_time }}**

---

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

为了检查正确性，可以先搜索数据库中的前5个向量，查看其最近邻居是否是自身。

```bash
[[  0 393 363  78]
 [  1 555 277 364]
 [  2 304 101  13]
 [  3 173  18 182]
 [  4 288 370 531]]

[[ 0.          7.17517328  7.2076292   7.25116253]
 [ 0.          6.32356453  6.6845808   6.79994535]
 [ 0.          5.79640865  6.39173603  7.28151226]
 [ 0.          7.27790546  7.52798653  7.66284657]
 [ 0.          6.76380348  7.29512024  7.36881447]]
```

实际搜索输出的结果类似于
```bash
[[ 381  207  210  477]
 [ 526  911  142   72]
 [ 838  527 1290  425]
 [ 196  184  164  359]
 [ 526  377  120  425]]

[[ 9900 10500  9309  9831]
 [11055 10895 10812 11321]
 [11353 11103 10164  9787]
 [10571 10664 10632  9638]
 [ 9628  9554 10036  9582]]
```

由于向量的第一个分量增加了附加值，数据集在 d 维空间的第一轴上被扩散。所以，前几个向量的邻居位于数据集的开头，而大约在 10000 附近的向量的邻居也在数据集的 10000 索引附近。

## Faster search (IVF)

```python
...

import faiss

nlist = 100
k = 4
quantizer = faiss.IndexFlatL2(d)  # the other index
# here we specify METRIC_L2, by default it performs inner-product search
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
index.train(xb)
index.add(xb)                  # add may be a bit slower as well

D, I = index.search(xq, k)     # actual search
print(I[-5:])                  # neighbors of the 5 last queries
index.nprobe = 10              # default nprobe is 1, try a few more
D, I = index.search(xq, k)
print(I[-5:])                  # neighbors of the 5 last queries
```

为了加快搜索速度，可以将数据集划分为多个片段。在 \( d \) 维空间中定义 Voronoi cells，每个数据库向量都会归属于其中一个细胞。

在搜索时，仅比较查询向量 \( x \) 所落入的细胞及其附近的一些相邻细胞中的数据库向量 \( y \)。

这一过程是通过 **IndexIVFFlat** 索引实现的。这种类型的索引需要一个训练阶段，该阶段可以在与数据库向量具有相同分布的任何向量集合上执行。在本例中，我们直接使用数据库向量进行训练。

IndexIVFFlat 还需要一个额外的索引，即 **quantizer**，用于将向量分配到 Voronoi 细胞中。每个细胞由一个**centroid**定义，确定一个向量属于哪个 Voronoi 细胞的过程，本质上是找到该向量在所有中心点中的最近邻居。这个任务通常由IndexFlatL2索引来完成。

搜索方法有两个关键参数：

- **nlist**：细胞的总数。
- **nprobe**：搜索时访问的细胞数量。  

搜索时间大致与探测的细胞数量成线性关系，同时还会有一部分由量化过程带来的额外开销。

当nprobe=1时，结果类似于：

```bash
[[ 9900 10500  9831 10808]
 [11055 10812 11321 10260]
 [11353 10164 10719 11013]
 [10571 10203 10793 10952]
 [ 9582 10304  9622  9229]]
```

当nprobe=10时，结果类似于：

```bash
[[ 9900 10500  9309  9831]
 [11055 10895 10812 11321]
 [11353 11103 10164  9787]
 [10571 10664 10632  9638]
 [ 9628  9554 10036  9582]]
```

得到了正确的结果。但是获得完美的结果只是因为该数据有很规律的分布，它在 x 轴上具有很强的分量，这使得它更容易处理。该参数始终是调整结果的速度和准确性之间的权衡的一种方式。设置 nprobe = nlist 给出的结果与暴力搜索相同。

## Lower memory footprint (PQ)

```python
...

nlist = 100
m = 8
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
# 8 specifies that each sub-vector is encoded as 8 bits
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)                                
index.train(xb)
index.add(xb)

D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
index.nprobe = 10              # make comparable with experiment above
D, I = index.search(xq, k)     # search
print(I[-5:])
```

之前看到的索引，**IndexFlatL2** 和 **IndexIVFFlat**，都会存储完整的向量。为了扩展到非常大的数据集，Faiss 提供了基于**乘积量化（Product Quantizer）**的有损压缩变体，以压缩存储的向量。

这些向量仍然存储在**Voronoi 单元**中，但其大小被减少到可配置的字节数 **m**（维度 **d** 必须是 **m** 的倍数）。

由于向量并不是以完全精确的形式存储的，因此搜索方法返回的距离值也是近似的。

运行结果如下：

```bash
[[   0  608  220  228]
 [   1 1063  277  617]
 [   2   46  114  304]
 [   3  791  527  316]
 [   4  159  288  393]]

[[ 1.40704751  6.19361687  6.34912491  6.35771513]
 [ 1.49901485  5.66632462  5.94188499  6.29570007]
 [ 1.63260388  6.04126883  6.18447495  6.26815748]
 [ 1.5356375   6.33165455  6.64519501  6.86594009]
 [ 1.46203303  6.5022912   6.62621975  6.63154221]]
```

可以看到最近邻被正确找到（即向量本身），但向量与自身的估算距离并不为 0（明显小于它与其他邻居的距离），这是由于有损压缩导致的。

这里将64 个 32 位浮点数压缩到8 字节，压缩比为32。

## Simplifying index construction

构建索引可能会变得复杂，Faiss 提供了工厂函数根据字符串来构造索引。

```python
index = faiss.index_factory(d, "IVF100,PQ8")
```

将 **PQ8** 替换为 **Flat**，即可获得 **IndexFlat**。当对输入向量进行**预处理（如 PCA）**时，工厂函数特别有用。例如，使用字符串 `"PCA32,IVF100,Flat"` 可以通过 **PCA 投影**将向量维度降至 **32D**。
