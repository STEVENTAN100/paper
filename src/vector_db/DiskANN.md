# DiskANN

## Introduction

DiskANN系列主要包含三篇论文：

- [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf)
- [FreshDiskANN: A Fast and Accurate Graph-Based  ANN Index for Streaming Similarity Search](https://arxiv.org/abs/2105.09613)
- [Filtered − DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf)

### SNG

## Algorithm

### GreedySearch

大多数基于图的近似最近邻搜索（ANN）算法的工作方式如下：在索引构建过程中，它们根据数据集 \( P \) 的几何属性构建图 \( G = (P, E) \)。在搜索时，对于查询向量 \( x_q \)，搜索采用自然的贪心或最优优先遍历，如算法1中的方式，在图 \( G \) 上进行。从某个指定的点 \( s \in P \) 开始，它们遍历图以逐步接近 \( x_q \)。

在SNG中，每个点 \( p \) 的外部邻居通过以下方式确定：初始化一个集合 \( S = P \setminus \{ p \} \)。只要 \( S \neq \emptyset \)，从 \( p \) 到 \( p^* \) 添加一条有向边，其中 \( p^* \) 是离 \( p \) 最近的点，从集合 \( S \) 中移除所有使得 \( d(p, p') > d(p^*, p') \)的点 \( p' \)。因此，贪心搜索（GreedySearch(s, \( x_p \), 1)）从任何 \( s \in P \) 开始都将收敛到 \( p \)，对于所有基点 \( p \in P \) 都成立。

![alg1](./img/diskann/alg1.png)

### RobustPrune

满足SNG性质的图都是适合贪心搜索搜索过程的良好候选图。然而，图的直径可能会相当大。例如，如果点在实数线的一维空间上线性排列，图的直径是 \( O(n) \)，其中每个点连接到两个邻居（一个在两端），这样的图满足SNG性质。搜索此类存储在磁盘上的图将导致在搜索路径中访问的顶点的邻居需要大量的顺序读取。

为了解决这个问题，希望确保查询距离在搜索路径的每个节点上按乘法因子 \( \alpha > 1 \) 递减，而不仅仅是像SNG性质那样递减。

![alg2](./img/diskann/alg2.png)

### Vamana Indexing

Vamana 以迭代的方式构建有向图 \( G \)。

1. 图 \( G \) 被初始化，使得每个顶点都有 \( R \) 个随机选择的外部邻居(在 \( R > \log n \) 时连接良好)。
2. 让 \( s \) 表示数据集 \( P \) 的中心点，它将作为搜索算法的起始节点。
3. 算法按随机顺序遍历 \( p \in P \) 的所有点，并在每一步更新图，使其更加适合贪心搜索（GreedySearch(s, \( x_p \), 1, L)）收敛到 \( p \)。

实际上，在对应点 \( p \) 的迭代中，

1. Vamana 在当前图 \( G \) 上运行 GreedySearch(s, \( x_p \), 1, L)，并将 \( V_p \) 设置为贪心搜索路径中所有访问过的点的集合。
2. 算法通过运行 RobustPrune(p, \( V_p \), \( \alpha \), \( R \)) 来更新图 \( G \)，以确定 \( p \) 的新外部邻居。
3. Vamana 更新图 \( G \)，通过为所有 \( p' \in N_{\text{out}}(p) \) 添加反向边（\( p', p \)）。这确保了在搜索路径和 \( p \) 之间的顶点连接，从而确保更新后的图会更适合贪心搜索（GreedySearch(s, \( x_p \), 1, L)）收敛到 \( p \)。
4. 添加这种形式的反向边（\( p', p \)）可能会导致 \( p' \) 的度数超过阈值，因此每当某个顶点 \( p' \) 的外度超过 \( R \) 的度数阈值时，图通过运行 RobustPrune(\( p' \), \( N_{\text{out}}(p') \), \( \alpha \), \( R \)) 来修改，其中 \( N_{\text{out}}(p') \) 是 \( p' \) 的现有外部邻居集合。

算法对数据集进行两次遍历，第一次遍历使用 \( \alpha = 1 \)，第二次使用用户定义的 \( \alpha \geq 1 \)。

![alg3](./img/diskann/alg3.png)

```admonish question
为什么要分成两次遍历，不在一次中直接设定一个大于1的 \( \alpha \) 值完成图的构建？
```

![example](./img/diskann/example.png)

## Design

### Index Design

在数据集 \( P \) 上运行 Vamana，并将结果存储在 SSD 上。在搜索时，每当算法1需要某个点 \( p \) 的外部邻居时， 从 SSD 中获取该点的信息。

然而，单纯存储包含十亿个100维空间中的向量数据远超过工作站的RAM！这引出了两个问题：

1. 如何构建一个包含十亿个点的图？
2. 如果不能在内存中存储向量数据，如何在算法1的搜索时执行查询点与候选点之间的距离比较？

**问题一**

通过聚类技术（如k-means）将数据划分成多个较小的分片，为每个分片建立一个单独的索引，并仅在查询时将查询路由到几个分片。然而，这种方法会因为查询需要路由到多个分片而导致搜索延迟增加和吞吐量减少。

与其在查询时将查询路由到多个分片，不如将每个基础点发送到多个相邻的中心以获得重叠的聚类。事实上，我们将一个十亿点的数据集通过k-means划分为k个聚类（k=40，通常ℓ=2就足够），然后将每个基础点分配给ℓ个最近的中心。接着，我们为分配给每个聚类的点建立Vamana索引（这些点现在只有约Nℓ/k个点，可以在内存中建立索引），最后通过简单的边缘合并将所有不同的图合并为一个单一的图。

**问题二**

将每个数据库点 \( p \in P \) 的压缩向量 \( \tilde{x}_p \) 存储在主存中，同时将图存储在SSD上。使用Product Quantization将数据和查询点编码为短代码在查询时高效地计算近似距离 \( d(\tilde{x}_p, x_q) \)。尽管在搜索时只使用压缩数据，但是Vamana在构建图索引时使用的是“全精度坐标”，因此能够高效地引导搜索到图的正确区域，

### Index Layout

将所有数据点的压缩向量存储在内存中，并将图以及全精度向量存储在SSD上。在磁盘中，对于每个点 \( i \)，我们存储其全精度向量 \( x_i \)，并跟随其\( \leq R \) 个邻居的标识。如果一个节点的度小于 \( R \)，我们用零填充，以便计算数据在磁盘中对应点 \( i \) 的偏移量变得简单，并且不需要在内存中存储偏移量。

### Beam Search

运行算法1，按需从SSD获取邻居信息 \(N_{\text{out}}(p^*)\)。这需要很多次SSD的往返（每次往返仅需几百微秒），从而导致更高的延迟。为了减少往返SSD的次数（以便顺序地获取邻居），而不过多的增加计算量，一次性获取一个较小数量 \(W\)（例如4、8）最近的点的邻居，并将 \(\mathcal{L}\) 更新为这一小步中的前 \(L\) 个最优点。从SSD获取少量随机扇区的时间几乎与获取一个扇区的时间相同。

- 如果 \(W = 1\)，这种搜索类似于正常的贪婪搜索。
- 如果 \(W\) 太大，比如16或更多，则计算和SSD带宽可能会浪费。

### Caching Frequently Visited Vertices

为了进一步减少每个查询的磁盘访问次数，缓存与一部分顶点相关的数据，这些顶点可以基于已知的查询分布来选择，或者通过缓存所有距离起始点 \( s \) 为 \( C = 3 \) 或 \( C = 4 \) 跳的顶点来选择。

由于索引图中与距离 \( C \) 相关的节点数量随着 \( C \) 增加呈指数增长，因此较大的 \( C \) 值会导致过大的内存占用。

### Implicit Re-Ranking Using Full-Precision Vectors

由于PQ是一种有损压缩方法，因此使用基于PQ的近似距离计算出的与查询最接近的 \( k \) 个候选点之间存在差距。为了解决这个问题，使用存储在磁盘上与每个点相邻的全精度坐标。

事实上，在搜索过程中检索一个点的邻域时，也可以在不增加额外磁盘读取的情况下获取该点的全精度坐标。这是因为读取4KB对齐的磁盘地址到内存的成本不高于读取512B，而顶点的邻域（对于度为128的图，邻域大小为 \( 4 \times 128 \) 字节）和全精度坐标可以存储在同一磁盘扇区中。

因此，随着BeamSearch加载搜索前沿的邻域，它还会缓存在搜索过程中**访问的所有节点**的全精度坐标，而不需要额外的SSD读取操作。这使得我们能够基于全精度向量返回前 \( k \) 个候选点。

## Summary

### Comparison of Vamana with HNSW and NSG

Vamana 与 HNSW 和 NSG 非常相似。这三种算法都在数据集 \( P \) 上进行迭代，并使用 GreedySearch(s, \( x_p \), 1, L) 和 RobustPrune(p, \( V \), \( \alpha \), \( R \)) 的结果来确定 \( p \) 的邻居。这些算法之间存在一些重要的差异。

- 最关键的是，HNSW 和 NSG 都没有可调参数 \( \alpha \)，而且隐式使用 \( \alpha = 1 \)。(这是使 Vamana 在图的度数和直径之间实现更好折衷的主要因素)
- 虽然 HNSW 将候选集 \( V \) 设置为贪心搜索（GreedySearch(s, \( p \), 1, L)）的修剪过程的最终候选结果集，Vamana 和 NSG 将 \( V \) 设置为贪心搜索（GreedySearch(s, \( p \), 1, L)）访问的所有顶点的集合。从直觉上讲，这个特征有助于 Vamana 和 NSG 添加长距离边，而 HNSW 仅通过添加局部边来构建图的层次结构，进而构建嵌套的图序列。
- NSG 将数据集上的图初始化为近似 \( K \)-最近邻图，这是一个时间和内存密集型的步骤，而 HNSW 和 Vamana 则有更简单的初始化过程，前者从一个空图开始，Vamana 从一个随机图开始。
- Vamana 进行两次遍历数据集，而 HNSW 和 NSG 只进行一次遍历，动机是基于我们观察到的第二次遍历能提高图的质量。

> - [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf)
> - [FreshDiskANN: A Fast and Accurate Graph-Based  ANN Index for Streaming Similarity Search](https://arxiv.org/abs/2105.09613)
> - [Filtered − DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf)
