# Basics

全文字数: **{{ #word_count }}**

阅读时间: **{{ #reading_time }}**

## MetricType and distances

### METRIC_L2

平方欧几里得（L2）距离（由于平方根是单调的，没有做平方根运算），此度量对数据的旋转不变（正交矩阵变换）。

### METRIC_INNER_PRODUCT

查询向量的范数不会影响结果的排名（数据库向量的范数确实重要）。其本身不是余弦相似度，除非向量已经归一化。

**如何对余弦相似度进行向量索引？**向量x和y之间的余弦相似度定义为：

\[
\text{cos}(x, y) = \frac{\langle x, y \rangle}{|x| \times |y|}
\]

它是相似度，而不是距离，因此通常会搜索相似度较大的向量。

通过预先归一化查询和数据库向量，可以将问题映射回最大内积搜索。执行此操作的方法：

- 使用**METRIC_INNER_PRODUCT**构建索引
- 在将向量添加到索引之前先归一化向量（在Python中使用**faiss.normalize_L2**）
- 在搜索之前先归一化向量

<!-- 请注意，这等同于使用带有**METRIC_L2**的索引，除了归一化向量的距离与\(|x - y|^2 = 2 - 2 \times \langle x, y \rangle\) 相关。 -->

## Clustering, PCA, Quantization

**Faiss 具有非常高效的k-means 聚类、PCA、PQ 编解码实现：。**

### Clustering

Faiss 提供了一个高效的 k-means 实现。将存储在给定 2D 张量 \( x \) 中的一组向量进行聚类，操作如下：

```python
ncentroids = 1024
niter = 20
verbose = True
d = x.shape[1]
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
kmeans.train(x)
```

最终的质心保存在 `kmeans.centroids` 中。

目标函数的值（在 k-means 中是总平方误差）在每次迭代中的变化存储在变量 `kmeans.obj` 中，更详细的统计信息存储在 `kmeans.iteration_stats` 中。

要在 GPU 上运行，在 Kmeans 构造函数中添加选项 `gpu=True`。（使用所有 GPU）或 `gpu=3`（使用 3 个 GPU）。

Kmeans 对象主要是 C++ 中 `Clustering` 对象的一层，所有该对象的字段可以通过构造函数进行设置。字段包括：

- `nredo`：运行聚类的次数，并保留最佳质心（根据聚类目标选择）
- `verbose`：使聚类过程更加详细
- `spherical`：执行球形 k-means —— 每次迭代后，质心会进行 L2 归一化
- `int_centroids`：将质心坐标四舍五入为整数
- `update_index`：每次迭代后是否重新训练索引？
- `min_points_per_centroid` / `max_points_per_centroid`：收到警告，表示训练集已被子采样
- `seed`：随机数生成器的种子

### Asignment

要计算从一组向量 \( x \) 到聚类质心的映射，在 k-means 训练完成后，可以使用：

```python
D, I = kmeans.index.search(x, 1)
```

将返回 \( x \) 中每个行向量的最近质心，\( I \) 包含这些质心的索引，\( D \) 包含平方的 L2 距离。

对于反向操作，例如查找 \( x \) 中距离已计算质心最近的 15 个点，必须使用新的索引：

```python
index = faiss.IndexFlatL2(d)
index.add(x)
D, I = index.search(kmeans.centroids, 15)
```

其中，大小为 \( (ncentroids, 15) \) 的 \( I \) 包含每个质心的最近邻。

### PCA

将 40D 向量降维到 10D。

```python
# 随机训练数据
mt = np.random.rand(1000, 40).astype('float32')
mat = faiss.PCAMatrix(40, 10)
mat.train(mt)
assert mat.is_trained
tr = mat.apply(mt)

# 打印此结果以展示 tr 的列的幅度在减少
print((tr ** 2).sum())
```

### Quantizers

量化器对象继承自 **Quantizer**，提供了三种常见的方法（见 `impl/Quantizer.h`）：

- **train**：在一组向量矩阵上训练量化器。
- **compute_codes** 和 **decode**：编码器和解码器。编码器通常是有损的，并返回每个输入向量的 **uint8** 类型代码矩阵。
- **get_DistanceComputer**：这是一个返回 **DistanceComputer** 对象的方法。

**Quantizer** 对象的状态是训练的结果（代码书表或归一化系数）。字段 **code_size** 表示量化器生成的每个代码的字节数。

支持的量化器类型有：

- **ScalarQuantizer**：将每个向量分量单独量化为线性范围内的值。
- **ProductQuantizer**：对子向量进行向量量化。
- **AdditiveQuantizer**：将一个向量编码为代码书条目的和，详情见 **Additive Quantizers**。Additive 量化器可以通过多种方式进行训练，因此有子类 **ResidualQuantizer**、**LocalSearchQuantizer** 和 **ProductAdditiveQuantizer**。

有趣的是，每个量化器都是前一个量化器的超集。

每个量化器都有一个对应的索引类型，它还存储一组量化向量。

| 量化器类               | 平面索引类                | IVF 索引类                |
|--------------------|------------------------|------------------------|
| ScalarQuantizer    | IndexScalarQuantizer   | IndexIVFScalarQuantizer |
| ProductQuantizer   | IndexPQ                | IndexIVFPQ              |
| AdditiveQuantizer  | IndexAdditiveQuantizer | IndexIVFAdditiveQuantizer |
| ResidualQuantizer  | IndexResidualQuantizer | IndexIVFResidualQuantizer |
| LocalSearchQuantizer | IndexLocalSearchQuantizer | IndexIVFLocalSearchQuantizer |

另外，除了 **ScalarQuantizer** 的索引，所有索引都有 **FastScan** 版本
<!-- 详见 **Fast accumulation of PQ and AQ codes**。 -->

一些量化器返回一个 **DistanceComputer** 对象。它的目的是在一个向量与许多代码进行比较时，高效地计算向量与代码之间的距离。在这种情况下，通常可以预先计算一组表，以便在压缩域中直接计算距离。

因此，**DistanceComputer** 提供以下方法：

- **set_query**：设置当前未压缩的向量进行比较。
- **distance_to_code**：计算与给定代码的实际距离。

**ProductQuantizer** 对象可用于将向量编码或解码为代码：

```python
d = 32  # 数据维度
cs = 4  # 代码大小（字节）

# 训练集
nt = 10000
xt = np.random.rand(nt, d).astype('float32')

# 编码数据集（可以与训练集相同）
n = 20000
x = np.random.rand(n, d).astype('float32')

pq = faiss.ProductQuantizer(d, cs, 8)
pq.train(xt)

# 编码
codes = pq.compute_codes(x)

# 解码
x2 = pq.decode(codes)

# 计算重建误差
avg_relative_error = ((x - x2)**2).sum() / (x ** 2).sum()
```

**标量量化器**的工作方式类似：

```python
d = 32  # 数据维度

# 训练集
nt = 10000
xt = np.random.rand(nt, d).astype('float32')

# 编码数据集（可以与训练集相同）
n = 20000
x = np.random.rand(n, d).astype('float32')

# QT_8bit 为每个维度分配 8 位（QT_4bit 也可以使用）
sq = faiss.ScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
sq.train(xt)

# 编码
codes = sq.compute_codes(x)

# 解码
x2 = sq.decode(codes)

# 计算重建误差
avg_relative_error = ((x - x2)**2).sum() / (x ** 2).sum()
```

## Guidelines to choose an index

1. 少量搜索？
   - 只进行少量搜索（例如 1000-10000 次），那么索引构建时间将无法通过搜索时间来摊销。此时，直接计算是最有效的选择。通过 "Flat" 索引来完成的。如果整个数据集无法放入 RAM 中，可以一个接一个地构建小的索引，并组合搜索结果。
2. 需要精确的结果？
   - 唯一可以保证精确结果的索引是 IndexFlatL2 或 IndexFlatIP。它为其他索引提供了结果的基准。它不支持`add_with_ids`，只支持顺序添加.

