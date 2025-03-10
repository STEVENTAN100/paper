# bench

## HNSW

> faiss/benchs/bench_hnsw.py

本程序用于对比不同索引算法在 SIFT1M 数据集上的性能表现，帮助用户评估搜索速度与准确率。测试方法包括 HNSW、IVF 等多种常见索引类型。

### 使用步骤

1. 基本命令格式

    ```bash
    python 程序名.py [k值] [测试方法1] [测试方法2]...
    ```

2. 参数说明
   - **k值** (必需): 指定每个查询要返回的最近邻数量
   - **测试方法** (可选): 指定要运行的算法测试，支持以下选项：
     - hnsw (HNSW 平面索引)
     - hnsw_sq (量化HNSW)
     - ivf (IVF 平面索引)
     - ivf_hnsw_quantizer (HNSW量化的IVF)
     - kmeans (K均值基准测试)
     - kmeans_hnsw (HNSW辅助的K均值)
     - nsg (NSG 图索引)
  
    *若不指定测试方法，默认运行所有测试*

3. 运行示例

    ```bash
    # 返回10个最近邻，测试HNSW和IVF
    python demo.py 10 hnsw ivf

    # 返回5个最近邻，测试所有方法
    python demo.py 5
    ```

4. 输出说明
    程序运行后会显示每个测试方法的以下指标：
    - **ms per query**: 每个查询的平均耗时（毫秒）
    - **R@1**: 前1个结果的召回率（准确率）
    - **missing rate**: 无效结果比例（-1表示未找到）

    示例输出：

    ```bash
    Testing HNSW Flat
    add...
    search...
    efSearch 16 bounded_queue True  0.153 ms per query, R@1 0.9210, missing rate 0.0000
    efSearch 16 bounded_queue False 0.162 ms per query, R@1 0.9210, missing rate 0.0000
    ...
    ```

```admonish note
在代码中，`missing rate` 是通过以下方式计算的：

`missing_rate = (I == -1).sum() / float(k * nq)`

- **`I`**: 是搜索结果的索引矩阵，形状为 `(nq, k)`，其中 `nq` 是查询的数量，`k` 是每个查询返回的最近邻数量。
  - 如果某个查询的某个最近邻未被找到，`I` 中对应的值会被设置为 `-1`。
- **`(I == -1).sum()`**: 统计所有查询结果中未找到的最近邻数量（即 `I` 中值为 `-1` 的元素总数）。
- **`k * nq`**: 表示理论上应该返回的总结果数量（每个查询返回 `k` 个结果，共有 `nq` 个查询）。
- **`missing_rate`**: 未找到的结果占总结果的比例。

直观上看，`missing rate`表现了要搜索k个最近邻但是并没有找到那么多的最近邻的情况。

```

### 运行结果

以下是对实验结果的详细分析，按测试方法分类总结性能指标（搜索速度、召回率、缺失率）和参数影响：

#### HNSW Flat (分层可导航小世界图)

| efSearch | Bounded Queue | 耗时/查询 | R@1   | 缺失率 |
|----------|---------------|-----------|-------|--------|
| 16       | True          | 0.006ms   | 0.9125| 0%     |
| 16       | False         | 0.006ms   | 0.9124| 0%     |
| 32       | True          | 0.010ms   | 0.9620| 0%     |
| 32       | False         | 0.010ms   | 0.9620| 0%     |
| 64       | True          | 0.017ms   | 0.9868| 0%     |
| 128      | True          | 0.029ms   | 0.9966| 0%     |
| 256      | True          | 0.055ms   | 0.9992| 0%     |

- **速度与召回率的权衡**：  
- `efSearch` 增大时，召回率显著提升（从 0.9125 到 0.9992），但搜索耗时线性增加（从 0.006ms 到 0.055ms）。
- **推荐参数**：  
  - 高精度场景：`efSearch=256`（R@1=99.92%），耗时 0.055ms/查询。  
  - 平衡场景：`efSearch=64`（R@1=98.68%），耗时 0.017ms/查询。
- **Bounded Queue 影响**：  
- Bounded Queue 开启与否对召回率和速度几乎无影响，可默认开启以节省内存。

#### HNSW with Scalar Quantizer (标量量化)

| efSearch | 耗时/查询 | R@1   | 缺失率 |
|----------|-----------|-------|--------|
| 16       | 0.002ms   | 0.7993| 0%     |
| 32       | 0.003ms   | 0.8877| 0%     |
| 64       | 0.006ms   | 0.9453| 0%     |
| 128      | 0.010ms   | 0.9780| 0%     |
| 256      | 0.017ms   | 0.9932| 0%     |

- **量化对性能的影响**：  
  - 相比 HNSW Flat，量化后搜索速度更快（`efSearch=256` 时 0.017ms vs 0.055ms），但召回率下降（99.32% vs 99.92%）。
  - **适用场景**：  
    - 内存敏感场景：量化可减少内存占用，但需接受轻微精度损失。
- **参数选择**：  
  - 若需接近原始精度，选择 `efSearch=256`（R@1=99.32%）。

#### IVF Flat (倒排文件索引)

| nprobe | 耗时/查询 | R@1   | 缺失率 |
|--------|-----------|-------|--------|
| 1      | 0.004ms   | 0.4139| 3.0%   |
| 4      | 0.005ms   | 0.6368| 0%     |
| 16     | 0.014ms   | 0.8322| 0%     |
| 64     | 0.046ms   | 0.9562| 0%     |
| 256    | 0.132ms   | 0.9961| 0%     |

- **nprobe 的重要性**：  
  - `nprobe=1` 时缺失率 3%，召回率仅 41.39%，表明搜索覆盖的聚类中心不足。
  - `nprobe=16` 时召回率 83.22%，耗时 0.014ms，性价比最高。
- **与 HNSW 对比**：  
  - 相同召回率下（如 99%），IVF 耗时更高（0.132ms vs HNSW 的 0.055ms）。

#### IVF with HNSW Quantizer

| nprobe | 耗时/查询 | R@1   | 缺失率 |
|--------|-----------|-------|--------|
| 1      | 0.003ms   | 0.4095| 3.16%  |
| 4      | 0.005ms   | 0.6324| 0%     |
| 16     | 0.013ms   | 0.8291| 0%     |
| 64     | 0.041ms   | 0.9540| 0%     |
| 256    | 0.136ms   | 0.9922| 0%     |

- **HNSW Quantizer 的影响**：  
  - 性能与传统 IVF 几乎一致，但训练速度可能更快（代码中未展示具体时间）。
  - 仍需设置 `nprobe≥16` 以避免高缺失率。

#### KMeans 聚类测试

- **Baseline KMeans**：  
  - 目标函数值 `3.8508e+10`，聚类不均衡度 `1.227`。
- **HNSW-Assisted KMeans**：  
  - 目标函数值略高（`3.85126e+10`），但训练时间显著减少（24.75s vs 53.52s）。

- **HNSW 加速效果**：  
  - HNSW 辅助的 KMeans 将训练时间减少 50%+，适合大规模数据聚类。

#### NSG (导航小世界图)

| search_L | 耗时/查询 | R@1   | 缺失率 |
|----------|-----------|-------|--------|
| -1       | 0.004ms   | 0.8150| 0%     |
| 16       | 0.006ms   | 0.8852| 0%     |
| 32       | 0.009ms   | 0.9533| 0%     |
| 64       | 0.015ms   | 0.9832| 0%     |
| 128      | 0.027ms   | 0.9959| 0%     |
| 256      | 0.050ms   | 0.9994| 0%     |

- **search_L 的影响**：  
  - `search_L=256` 时召回率 99.94%，与 HNSW Flat（99.92%）相当，但耗时稍高（0.050ms vs 0.055ms）。
  - **优势**：NSG 在中等参数下（如 `search_L=64`）即可达到 98.32% 召回率，适合对精度和速度均衡要求较高的场景。

#### 综合对比与推荐

| 索引类型          | 最佳参数          | 耗时/查询 | R@1   | 适用场景                     |
|-------------------|-------------------|-----------|-------|------------------------------|
| **HNSW Flat**     | efSearch=64       | 0.017ms   | 98.68%| 高精度实时搜索               |
| **HNSW SQ**       | efSearch=256      | 0.017ms   | 99.32%| 内存敏感场景                 |
| **IVF Flat**      | nprobe=64         | 0.046ms   | 95.62%| 高吞吐量批量查询             |
| **NSG**           | search_L=64       | 0.015ms   | 98.32%| 均衡精度与速度               |
| **IVF+HNSQ Quant**| nprobe=64         | 0.041ms   | 95.40%| 需要快速训练的 IVF 场景       |

- **精度优先**：选择 **HNSW Flat（efSearch=256）** 或 **NSG（search_L=256）**。
- **速度优先**：选择 **HNSW SQ（efSearch=16）** 或 **IVF Flat（nprobe=16）**。
- **内存敏感**：使用 **HNSW SQ** 或 **IVF 索引**。
- **训练时间敏感**：尝试 **HNSW-Assisted KMeans** 加速聚类。

```admonish success collapsible=true, title='实验输出'
    
    (faiss_env) gpu@gpu-node:~/faiss/faiss/benchs$ python bench_hnsw.py 10
    load data
    Testing HNSW Flat
    add
    hnsw_add_vertices: adding 1000000 elements on top of 0 (preset_levels=0)
    max_level = 4
    Adding 1 elements at level 4
    Adding 26 elements at level 3
    Adding 951 elements at level 2
    Adding 30276 elements at level 1
    Adding 968746 elements at level 0
    Done in 10194.740 ms
    search
    efSearch 16 bounded queue True     0.006 ms per query, R@1 0.9125, missing rate 0.0000
    efSearch 16 bounded queue False            0.006 ms per query, R@1 0.9124, missing rate 0.0000
    efSearch 32 bounded queue True     0.010 ms per query, R@1 0.9620, missing rate 0.0000
    efSearch 32 bounded queue False            0.010 ms per query, R@1 0.9620, missing rate 0.0000
    efSearch 64 bounded queue True     0.017 ms per query, R@1 0.9868, missing rate 0.0000
    efSearch 64 bounded queue False            0.017 ms per query, R@1 0.9868, missing rate 0.0000
    efSearch 128 bounded queue True            0.029 ms per query, R@1 0.9966, missing rate 0.0000
    efSearch 128 bounded queue False           0.030 ms per query, R@1 0.9966, missing rate 0.0000
    efSearch 256 bounded queue True            0.055 ms per query, R@1 0.9992, missing rate 0.0000
    efSearch 256 bounded queue False           0.055 ms per query, R@1 0.9992, missing rate 0.0000
    Testing HNSW with a scalar quantizer
    training
    add
    hnsw_add_vertices: adding 1000000 elements on top of 0 (preset_levels=0)
    max_level = 5
    Adding 1 elements at level 5
    Adding 15 elements at level 4
    Adding 194 elements at level 3
    Adding 3693 elements at level 2
    Adding 58500 elements at level 1
    Adding 937597 elements at level 0
    Done in 3322.349 ms
    search
    efSearch 16        0.002 ms per query, R@1 0.7993, missing rate 0.0000
    efSearch 32        0.003 ms per query, R@1 0.8877, missing rate 0.0000
    efSearch 64        0.006 ms per query, R@1 0.9453, missing rate 0.0000
    efSearch 128       0.010 ms per query, R@1 0.9780, missing rate 0.0000
    efSearch 256       0.017 ms per query, R@1 0.9932, missing rate 0.0000
    Testing IVF Flat (baseline)
    training
    Training level-1 quantizer
    Training level-1 quantizer on 100000 vectors in 128D
    Training IVF residual
    IndexIVF: no residual training
    add
    IndexIVFFlat::add_core: added 1000000 / 1000000 vectors
    search
    nprobe 1           0.004 ms per query, R@1 0.4139, missing rate 0.0300
    nprobe 4           0.005 ms per query, R@1 0.6368, missing rate 0.0000
    nprobe 16          0.014 ms per query, R@1 0.8322, missing rate 0.0000
    nprobe 64          0.046 ms per query, R@1 0.9562, missing rate 0.0000
    nprobe 256         0.132 ms per query, R@1 0.9961, missing rate 0.0000
    Testing IVF Flat with HNSW quantizer
    training
    Training level-1 quantizer
    Training L2 quantizer on 100000 vectors in 128D
    Adding centroids to quantizer
    Training IVF residual
    IndexIVF: no residual training
    add
    IndexIVFFlat::add_core: added 1000000 / 1000000 vectors
    search
    nprobe 1           0.003 ms per query, R@1 0.4095, missing rate 0.0316
    nprobe 4           0.005 ms per query, R@1 0.6324, missing rate 0.0000
    nprobe 16          0.013 ms per query, R@1 0.8291, missing rate 0.0000
    nprobe 64          0.041 ms per query, R@1 0.9540, missing rate 0.0000
    nprobe 256         0.136 ms per query, R@1 0.9922, missing rate 0.0000
    Performing kmeans on sift1M database vectors (baseline)
    Clustering 1000000 points in 128D to 16384 clusters, redo 1 times, 10 iterations
    Preprocessing in 0.15 s
    Iteration 9 (53.52 s, search 53.23 s): objective=3.8508e+10 imbalance=1.227 nsplit=0        
    Performing kmeans on sift1M using HNSW assignment
    Clustering 1000000 points in 128D to 16384 clusters, redo 1 times, 10 iterations
    Preprocessing in 0.15 s
    Iteration 9 (24.75 s, search 24.04 s): objective=3.85126e+10 imbalance=1.227 nsplit=0       
    Testing NSG Flat
    add
    IndexNSG::add 1000000 vectors
    Build knn graph with NNdescent S=10 R=100 L=114 niter=10
    Parameters: K=64, S=10, R=100, L=114, iter=10
    Iter: 0, recall@64: 0.000000
    Iter: 1, recall@64: 0.002813
    Iter: 2, recall@64: 0.024688
    Iter: 3, recall@64: 0.117656
    Iter: 4, recall@64: 0.334219
    Iter: 5, recall@64: 0.579063
    Iter: 6, recall@64: 0.759375
    Iter: 7, recall@64: 0.867188
    Iter: 8, recall@64: 0.928594
    Iter: 9, recall@64: 0.959063
    Added 1000000 points into the index
    Check the knn graph
    nsg building
    NSG::build R=32, L=64, C=132
    Degree Statistics: Max = 32, Min = 1, Avg = 20.055180
    Attached nodes: 0
    search
    search_L -1        0.004 ms per query, R@1 0.8150, missing rate 0.0000
    search_L 16        0.006 ms per query, R@1 0.8852, missing rate 0.0000
    search_L 32        0.009 ms per query, R@1 0.9533, missing rate 0.0000
    search_L 64        0.015 ms per query, R@1 0.9832, missing rate 0.0000
    search_L 128       0.027 ms per query, R@1 0.9959, missing rate 0.0000
    search_L 256       0.050 ms per query, R@1 0.9994, missing rate 0.0000
```

### 运行结果（WSL）

#### HNSW Flat 性能

| **参数**               | **构建时间 (ms)** | **搜索时间 (ms/query)** | **R@1** | **缺失率** |
|------------------------|-------------------|-------------------------|---------|------------|
| efSearch=16 (bounded)  | 28,095.8          | 0.015                   | 0.9069  | 0.0000     |
| efSearch=16 (unbounded)| 28,095.8          | 0.015                   | 0.9069  | 0.0000     |
| efSearch=32 (bounded)  | 28,095.8          | 0.023                   | 0.9614  | 0.0000     |
| efSearch=32 (unbounded)| 28,095.8          | 0.025                   | 0.9614  | 0.0000     |
| efSearch=64 (bounded)  | 28,095.8          | 0.043                   | 0.9868  | 0.0000     |
| efSearch=64 (unbounded)| 28,095.8          | 0.041                   | 0.9868  | 0.0000     |
| efSearch=128 (bounded) | 28,095.8          | 0.069                   | 0.9964  | 0.0000     |
| efSearch=256 (bounded) | 28,095.8          | 0.126                   | 0.9995  | 0.0000     |

- **搜索效率与召回率正相关**：`efSearch` 值越大，搜索时间增加，但召回率显著提升（如 `efSearch=256` 时 R@1 达 0.9995）。
- **队列类型影响小**：`bounded` 与 `unbounded` 队列对搜索性能无明显差异。

#### HNSW 标量量化（SQ）性能

| **参数**       | **构建时间 (ms)** | **搜索时间 (ms/query)** | **R@1** | **缺失率** |
|----------------|-------------------|-------------------------|---------|------------|
| efSearch=16    | 12,376.8          | 0.007                   | 0.7884  | 0.0000     |
| efSearch=32    | 12,376.8          | 0.011                   | 0.8827  | 0.0000     |
| efSearch=64    | 12,376.8          | 0.020                   | 0.9419  | 0.0000     |
| efSearch=128   | 12,376.8          | 0.033                   | 0.9751  | 0.0000     |
| efSearch=256   | 12,376.8          | 0.055                   | 0.9912  | 0.0000     |

- **量化显著加速构建**：构建时间缩短至 12,376.8 ms（相比 HNSW Flat 的 28,095.8 ms）。
- **召回率略降**：相同 `efSearch` 下，量化后 R@1 低于原始 HNSW（如 `efSearch=256` 时 R@1=0.9912 vs. 0.9995）。

#### IVF Flat 基准性能

| **参数**    | **搜索时间 (ms/query)** | **R@1** | **缺失率** |
|-------------|-------------------------|---------|------------|
| nprobe=1    | 0.022                   | 0.4139  | 0.0300     |
| nprobe=4    | 0.018                   | 0.6368  | 0.0000     |
| nprobe=16   | 0.035                   | 0.8322  | 0.0000     |
| nprobe=64   | 0.103                   | 0.9562  | 0.0000     |
| nprobe=256  | 0.332                   | 0.9961  | 0.0000     |

- **需高 `nprobe` 保证召回率**：`nprobe=256` 时 R@1 接近 HNSW，但搜索时间显著增加（0.332 ms vs. HNSW 的 0.126 ms）。

#### NSG Flat 性能

| **参数**       | **搜索时间 (ms/query)** | **R@1** | **缺失率** |
|----------------|-------------------------|---------|------------|
| search_L=-1    | 0.008                   | 0.8152  | 0.0000     |
| search_L=16    | 0.011                   | 0.8849  | 0.0000     |
| search_L=32    | 0.017                   | 0.9528  | 0.0000     |
| search_L=64    | 0.032                   | 0.9843  | 0.0000     |
| search_L=128   | 0.051                   | 0.9959  | 0.0000     |
| search_L=256   | 0.086                   | 0.9993  | 0.0000     |

- **高效搜索与高召回率**：在 `search_L=256` 时，R@1 达 0.9993，且搜索时间（0.086 ms）优于 HNSW Flat 的 0.126 ms。

#### 综合对比

| **索引类型**          | **构建时间 (ms)** | **最佳 R@1** | **对应搜索时间 (ms)** |
|-----------------------|-------------------|--------------|-----------------------|
| HNSW Flat             | 28,095.8          | 0.9995       | 0.126                 |
| HNSW SQ               | 12,376.8          | 0.9912       | 0.055                 |
| IVF Flat              | -                 | 0.9961       | 0.332                 |
| NSG Flat              | -                 | 0.9993       | 0.086                 |

1. **HNSW 系列**：  
   - 适合高召回率场景，但构建时间较长；量化版本（HNSW SQ）可显著加速构建，但牺牲少量召回率。
2. **IVF Flat**：  
   - 需要高 `nprobe` 达到高召回率，但搜索效率较低。
3. **NSG Flat**：  
   - 在搜索速度与召回率间取得最佳平衡（`search_L=256` 时 R@1=0.9993，搜索时间仅 0.086 ms）。

- 实时搜索：优先选择 **HNSW SQ** 或 **NSG Flat**；  
- 高精度需求：选择 **HNSW Flat** 或 **NSG Flat**；  
- 快速构建：选择 **HNSW SQ**。

```admonish success collapsible=true, title='实验输出'
    (faiss_env) root@Quaternijkon:~/faiss/benchs# python bench_hnsw.py 10
    load data
    Testing HNSW Flat
    add
    hnsw_add_vertices: adding 1000000 elements on top of 0 (preset_levels=0)
      max_level = 4
    Adding 1 elements at level 4
    Adding 26 elements at level 3
    Adding 951 elements at level 2
    Adding 30276 elements at level 1
    Adding 968746 elements at level 0
    Done in 28095.800 ms
    search
    efSearch 16 bounded queue True     0.015 ms per query, R@1 0.9069, missing rate 0.0000
    efSearch 16 bounded queue False            0.015 ms per query, R@1 0.9069, missing rate 0.0000
    efSearch 32 bounded queue True     0.023 ms per query, R@1 0.9614, missing rate 0.0000
    efSearch 32 bounded queue False            0.025 ms per query, R@1 0.9614, missing rate 0.0000
    efSearch 64 bounded queue True     0.043 ms per query, R@1 0.9868, missing rate 0.0000
    efSearch 64 bounded queue False            0.041 ms per query, R@1 0.9868, missing rate 0.0000
    efSearch 128 bounded queue True            0.069 ms per query, R@1 0.9964, missing rate 0.0000
    efSearch 128 bounded queue False           0.069 ms per query, R@1 0.9964, missing rate 0.0000
    efSearch 256 bounded queue True            0.126 ms per query, R@1 0.9995, missing rate 0.0000
    efSearch 256 bounded queue False           0.122 ms per query, R@1 0.9995, missing rate 0.0000
    Testing HNSW with a scalar quantizer
    training
    add
    hnsw_add_vertices: adding 1000000 elements on top of 0 (preset_levels=0)
      max_level = 5
    Adding 1 elements at level 5
    Adding 15 elements at level 4
    Adding 194 elements at level 3
    Adding 3693 elements at level 2
    Adding 58500 elements at level 1
    Adding 937597 elements at level 0
    Done in 12376.822 ms
    search
    efSearch 16        0.007 ms per query, R@1 0.7884, missing rate 0.0000
    efSearch 32        0.011 ms per query, R@1 0.8827, missing rate 0.0000
    efSearch 64        0.020 ms per query, R@1 0.9419, missing rate 0.0000
    efSearch 128       0.033 ms per query, R@1 0.9751, missing rate 0.0000
    efSearch 256       0.055 ms per query, R@1 0.9912, missing rate 0.0000
    Testing IVF Flat (baseline)
    training
    Training level-1 quantizer
    Training level-1 quantizer on 100000 vectors in 128D
    Training IVF residual
    IndexIVF: no residual training
    add
    IndexIVFFlat::add_core: added 1000000 / 1000000 vectors
    search
    nprobe 1           0.022 ms per query, R@1 0.4139, missing rate 0.0300
    nprobe 4           0.018 ms per query, R@1 0.6368, missing rate 0.0000
    nprobe 16          0.035 ms per query, R@1 0.8322, missing rate 0.0000
    nprobe 64          0.103 ms per query, R@1 0.9562, missing rate 0.0000
    nprobe 256         0.332 ms per query, R@1 0.9961, missing rate 0.0000
    Testing IVF Flat with HNSW quantizer
    training
    Training level-1 quantizer
    Training L2 quantizer on 100000 vectors in 128D
    Adding centroids to quantizer
    Training IVF residual
    IndexIVF: no residual training
    add
    IndexIVFFlat::add_core: added 1000000 / 1000000 vectors
    search
    nprobe 1           0.009 ms per query, R@1 0.4102, missing rate 0.0323
    nprobe 4           0.014 ms per query, R@1 0.6327, missing rate 0.0000
    nprobe 16          0.030 ms per query, R@1 0.8288, missing rate 0.0000
    nprobe 64          0.094 ms per query, R@1 0.9545, missing rate 0.0000
    nprobe 256         0.332 ms per query, R@1 0.9925, missing rate 0.0000
    Performing kmeans on sift1M database vectors (baseline)
    Clustering 1000000 points in 128D to 16384 clusters, redo 1 times, 10 iterations
      Preprocessing in 0.07 s
      Iteration 0 (36.15 s, search 36.12 s): objective=5.58803e+10 imbalance=1.737 nsplit=5  Iteration 1 (71.32 s, search 71.25 s): objective=4.11771e+10 imbalance=1.378 nsplit=0  Iteration 2 (107.25 s, search 107.15 s): objective=3.9914e+10 imbalance=1.301 nsplit=  Iteration 3 (142.60 s, search 142.47 s): objective=3.93716e+10 imbalance=1.269 nsplit  Iteration 4 (178.17 s, search 178.01 s): objective=3.9066e+10 imbalance=1.253 nsplit=  Iteration 5 (214.30 s, search 214.11 s): objective=3.8872e+10 imbalance=1.243 nsplit=  Iteration 6 (250.13 s, search 249.92 s): objective=3.87377e+10 imbalance=1.237 nsplit  Iteration 7 (285.57 s, search 285.33 s): objective=3.86397e+10 imbalance=1.232 nsplit  Iteration 8 (321.46 s, search 321.19 s): objective=3.85653e+10 imbalance=1.229 nsplit  Iteration 9 (357.11 s, search 356.81 s): objective=3.8508e+10 imbalance=1.227 nsplit=0       
    Performing kmeans on sift1M using HNSW assignment
    Clustering 1000000 points in 128D to 16384 clusters, redo 1 times, 10 iterations
      Preprocessing in 0.07 s
      Iteration 0 (11.13 s, search 10.96 s): objective=5.58887e+10 imbalance=1.737 nsplit=0  Iteration 1 (21.37 s, search 21.07 s): objective=4.11817e+10 imbalance=1.379 nsplit=2  Iteration 2 (31.33 s, search 30.89 s): objective=3.99504e+10 imbalance=1.301 nsplit=0  Iteration 3 (41.08 s, search 40.47 s): objective=3.93767e+10 imbalance=1.269 nsplit=2  Iteration 4 (50.91 s, search 50.17 s): objective=3.90675e+10 imbalance=1.253 nsplit=1  Iteration 5 (60.60 s, search 59.72 s): objective=3.88936e+10 imbalance=1.243 nsplit=0  Iteration 6 (70.48 s, search 69.46 s): objective=3.87889e+10 imbalance=1.236 nsplit=0  Iteration 7 (80.54 s, search 79.37 s): objective=3.8648e+10 imbalance=1.232 nsplit=1   Iteration 8 (90.33 s, search 89.03 s): objective=3.85683e+10 imbalance=1.229 nsplit=3  Iteration 9 (100.14 s, search 98.69 s): objective=3.85264e+10 imbalance=1.227 nsplit=1       
    Testing NSG Flat
    add
    IndexNSG::add 1000000 vectors
      Build knn graph with NNdescent S=10 R=100 L=114 niter=10
    Parameters: K=64, S=10, R=100, L=114, iter=10
    Iter: 0, recall@64: 0.000000
    Iter: 1, recall@64: 0.002969
    Iter: 2, recall@64: 0.024531
    Iter: 3, recall@64: 0.110781
    Iter: 4, recall@64: 0.332656
    Iter: 5, recall@64: 0.559375
    Iter: 6, recall@64: 0.751094
    Iter: 7, recall@64: 0.862188
    Iter: 8, recall@64: 0.922656
    Iter: 9, recall@64: 0.953906
    Added 1000000 points into the index
      Check the knn graph
      nsg building
    NSG::build R=32, L=64, C=132
    Degree Statistics: Max = 32, Min = 1, Avg = 20.056442
    Attached nodes: 0
    search
    search_L -1        0.008 ms per query, R@1 0.8152, missing rate 0.0000
    search_L 16        0.011 ms per query, R@1 0.8849, missing rate 0.0000
    search_L 32        0.017 ms per query, R@1 0.9528, missing rate 0.0000
    search_L 64        0.032 ms per query, R@1 0.9843, missing rate 0.0000
    search_L 128       0.051 ms per query, R@1 0.9959, missing rate 0.0000
    search_L 256       0.086 ms per query, R@1 0.9993, missing rate 0.0000
```
