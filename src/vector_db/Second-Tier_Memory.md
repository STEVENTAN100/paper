# Characterizing the Dilemma of Performance and Index Size in Billion-Scale Vector  Search and Breaking It with Second-Tier Memory

[Characterizing the Dilemma of Performance and Index Size in Billion-Scale Vector  Search and Breaking It with Second-Tier Memory](https://arxiv.org/abs/2405.03267)

全文字数：**{{ #word_count }}**

阅读时间：**{{ #reading_time }}**

## Background

**Second-tier memory** (such as remote DRAM/NVM connected via RDMA or CXL), with its fine-grained access granularity, offers an effective solution to address **the mismatch between search and access granularity**.

However:
- Existing indexes, primarily designed for SSDs, do not perform well on second-tier memory.
- Second-tier memory behaves more like storage, making its use as DRAM inefficient.

To address this, the paper developed a graph and cluster index tailored to the performance characteristics of second-tier memory. 
- Optimal performance
- Orders of magnitude smaller index amplification.

## High Performance and Low Index Size Dilemma

Increasing index size improves performance:  
- For graph indexes, adding more edges reduces the number of jumps (I/O ).  
- For cluster indexes, duplicating neighboring vectors across clusters minimizes the number of clusters searched.  

Conversely, reducing index size leads to performance degradation. An alternative to reduce index size without altering the algorithm is to **copy addresses instead of duplicating vectors**. However, this introduces additional small random reads, which significantly degrade performance on SSDs.

**Root cause**: **Workload patterns do not align with SSD requirements.**
- SSD pages are typically 4KB, while second-tier memory supports access at a finer granularity of 256B.  
- Both graph and cluster indexes typically perform reads of a few hundred bytes at a time.

### Graph Index

![Graph Index](./img/graph.png)

- Graph indexes capture the fine-grained relationships between vectors, enabling low read amplification by accessing only a few additional vectors beyond the required top-k.  
- However, their pointer-chasing access pattern leads to high latency and poor bandwidth utilization.

DiskANN assigns 32 edges to each vector by default, with each edge represented as a 4B vector ID. This means the size of the edge list can exceed the size of the raw data. For example, representing a vertex with 100B in SPACEV requires 128B to store its edges.  

Ideally, the space amplification caused by edges should be minimized. However, reducing the number of edges increases the traversal distance for query vectors.  

Even if we increase the number of edges, the load per graph traversal remains limited by the SSD access granularity (4KB). Exceeding this would drastically inflate the index size—up to 39 times the size of the original dataset. For instance, storing a node with 100B while using 3,996B for edges leads to excessive space usage.



### Cluster Index

![Cluster Index](./img/cluster.png)

- Efficient for storage, enabling **large bulk reads** of clusters.  
- Suffer from high **read amplification** due to redundant vector reads within clusters.  
- Replications lead to high **space amplification**.

Ideally, cluster indexes should have a negligible index size:  
- The indexing data for clusters is several orders of magnitude smaller than the total dataset.  
- Each cluster may span multiple blocks, eliminating the need for padding.  

To address accuracy loss caused by boundary issues, boundary vectors are duplicated across adjacent clusters, requiring additional storage for the index.  

To reduce excessive duplication, an alternative is to increase the number of clusters searched per query. Unfortunately, additional cluster searches degrade performance due to the extra vectors read.  

Empirically, increasing the index size can still improve throughput.

### Workloads Mismatch SSD Requirements

- Graph index requires fine-grained storage reads for prac-tical index sizes.which  conflicts with the requirement of using a sufficiently large I/O payload(4 KB) to efficiently utilize traditional storage devices like SSD.
- Cluster index requires irregular I/O for deduplication.instead of storing replicated vector data in other clusters, we store an address pointing to the original cluster. the vector address(8 B) is significantly smaller than the data(128–384 B). However, it implies that each replicated vector requires a separate small random read(100–384 B) to fetch the original vector. 

## Power of Second-tier Memory