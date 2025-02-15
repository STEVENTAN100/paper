# vector db

全文字数: **{{ #word_count }}**

阅读时间: **{{ #reading_time }}**

---

## Index

- 内存索引
    - brute-force
        - Flat
    - 倒排：
        - IVF_FLAT
        - IVF_SQ8
        - IVF_PQ
    - 树：
    - 哈希：
    - 图：
        - [HNSW](./HNSW.md)

- 磁盘索引
    - 图
        - [DiskANN](./DiskANN.md)
    - 簇
        - [SPANN](./SPANN.md)
        - SPFresh

| 索引算法 | 介质 | 优点 | 缺点 |
| --- | --- | --- | --- |
| Flat | 内存 | 精确 | 速度慢 |

### Brute-force

#### Flat

暴力搜索

- 保证精确度
- 效率低

适合在小型、百万级数据集上寻求完全精确的搜索结果。

### IVF

#### IVF_FLAT

使用倒排索引方法，结合聚类算法（如k-means），将高维空间中的向量划分为多个簇。每个簇包含相似向量，并且选取簇的中心作为代表向量。为每个簇创建倒排索引，并将每个向量映射到所属的簇。

查询时，系统只需关注相似簇，避免了遍历整个高维空间，从而显著降低了搜索复杂度。通过参数nprobe控制搜索时考虑的簇数，平衡查询精度与速度。

- IVF_FLAT将查询向量分配到距离最近的簇中心
- 然后在该簇内执行线性搜索，查找与查询向量相似的向量。

IVF_FLAT适用于需要在精度和查询速度之间取得平衡的场景，特别是在大规模高维数据集上，能够显著降低查询时间。它非常适合那些对精度要求较高，但可以容忍一定查询延迟的应用。

#### IVF_SQ8

IVF_SQ8 在 IVF_FLAT 基础上增加了量化步骤。IVF_SQ8通过标量量化（Scalar Quantization）将每个维度的 4 字节浮点数表示压缩为 1 字节整数表示。量化的过程是将原始的浮点数值映射到一个较小的整数范围。

- 通过量化将存储和计算资源的消耗大大降低
- 核心思想与 IVF_FLAT 类似

#### IVF_PQ

IVF_PQ 在 IVF_FLAT 基础上增加了 Product Quantization 步骤。将向量空间划分为更小的子空间并进行量化。

通过将向量划分为多个子空间，并对每个子空间进行独立的量化，生成一个代码本（codebook）。这样，原始的高维向量可以由多个子空间的量化表示组合而成，从而降低存储需求并加速检索。

在IVF_PQ中，乘积量化应用于IVF的聚类过程。每个簇的中心点会被进一步量化，原始的查询向量和数据向量在计算距离时，不是直接与每个簇中心进行计算，而是与每个子空间的量化中心进行计算。这种方法不仅降低了存储开销，还减少了计算距离时的运算量。

[PQ相关内容详见](./PQ.md)

### Graph-based

#### HNSW

HNSW（Hierarchical Navigable Small World Graph）是一种图索引算法，基于分层结构和小世界图理论，用于高效的近似最近邻搜索。它构建一个多层图结构，逐层精细化，提高高维数据集的搜索效率。

HNSW结合了跳表（Skip List）和小世界图（NSW）的技术：

- **跳表**：底层为有序链表，逐层稀疏化，查询从最上层开始，逐层向下查找。
- **NSW图**：每个节点连接若干相似节点，使用邻接列表，从入口节点开始通过边查找最接近查询向量的节点。

HNSW工作分为两个主要阶段：索引构建和查询过程:

- **索引构建**：HNSW构建多层图，每层提供不同精度和速度的搜索。最上层图提供粗粒度搜索，底层则提供更精细搜索。每个新节点连接若干近邻，形成局部小世界结构，保证图的稀疏性和高效性。
- **查询过程**：查询从最上层开始，逐层向下，通过图的边接近目标。到达底层后，HNSW使用局部搜索找到最接近查询向量的结果。

[HNSW相关内容详见](./HNSW.md)

#### DiskANN

### Cluster-based

#### SPANN

## Strategy

- 

## Graph

### Graph Structure

#### Delaunay Graphs

#### Relative Neighborhood Graphs (RNG)

#### Navigable Small-World Networks (NSW)

#### Randomized Neighborhood Graphs

## Dataset

> [向量索引的基本方法有哪些？](https://www.zhihu.com/org/zilliz-11/answers)