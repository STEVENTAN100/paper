# DiskANN

## 项目结构

### ./include

```admonish note collapsible=true, title=''

```

```admonish note collapsible=true, title='abstract_data_store.h'
这段代码定义了一个抽象基类 `AbstractDataStore`，用于表示和管理数据存储。它是一个模板类，适用于不同类型的数据（`data_t`）。这个类定义了如何加载、保存、操作和检索数据点，并支持一些高级功能，如数据预处理、距离计算和动态扩展存储空间。

1. **数据存储和加载**：
   - `load`：从文件中加载数据。
   - `save`：将数据保存到文件。
   - `resize`：调整数据存储的容量。

2. **数据处理**：
   - `populate_data`：将数据点填充到存储中，可以是从文件或指针中加载。
   - `extract_data_to_bin`：将数据从存储中提取到二进制文件。
   - `move_vectors` 和 `copy_vectors`：对存储中的向量进行移动或复制操作。

3. **查询与更新**：
   - `get_vector` 和 `set_vector`：检索和设置单个数据点。
   - `prefetch_vector`：预取数据点，以提高访问速度。
   - `preprocess_query`：对查询数据进行预处理。

4. **距离计算**：
   - `get_distance`：计算查询向量与存储中某个数据点之间的距离。支持多种重载方式，允许不同类型的查询。

5. **数据统计**：
   - `calculate_medoid`：计算数据集的“中点”，即距离所有点的平均值最接近的点。 

6. **存储容量管理**：
   - `expand` 和 `shrink`：扩展或收缩数据存储的容量，以适应不同的需求。

7. **对齐与距离函数**：
   - `get_aligned_dim`：返回对齐后的维度（用于某些距离度量）。
   - `get_dist_fn`：获取用于计算距离的函数。

`AbstractDataStore` 类定义了一个数据存储的抽象接口，用于处理数据的加载、保存、扩展、查询、距离计算等操作。这个类的设计非常灵活，允许不同的数据存储实现方式，可以用于支持不同类型的数据存储后端（如内存、磁盘、数据库等）。它为支持ANN（近似最近邻）算法中的数据存储和管理提供了基础设施。
```

```admonish note collapsible=true, title='abstract_graph_store.h'
这段代码定义了一个抽象类 `AbstractGraphStore`，用于表示和管理图数据存储。图数据结构通常用于存储连接性信息，例如图的邻接关系。

1. **图数据加载与存储：**
   - `load`：加载图数据，从指定路径读取图数据并返回一些有关图的元数据。
   - `store`：将图数据保存到指定路径。

2. **邻居管理：**
   - `get_neighbours`：获取指定节点的邻居列表。
   - `add_neighbour`：向指定节点添加邻居。
   - `clear_neighbours`：清除指定节点的所有邻居。
   - `swap_neighbours`：交换两个节点的邻居信息。
   - `set_neighbours`：设置指定节点的邻居列表。

3. **图容量和大小管理：**
   - `resize_graph`：调整图的大小，可能会扩展或收缩图。
   - `clear_graph`：清空图的数据。
   - `get_total_points`：获取图中的总节点数。
   - `get_max_observed_degree`：获取图中节点的最大度数（即一个节点连接的最大邻居数）。
   - `get_max_range_of_graph`：获取图的最大范围，通常与图的连接性或数据分布有关。

4. **保护性操作：**
   - `set_total_points`：内部函数，用于调整图的总节点数。

`AbstractGraphStore` 类是一个抽象接口，定义了图数据的存储、加载、管理和操作方法。它专注于存储图结构的数据，如节点与节点之间的连接关系，并提供操作这些连接（邻居）的基本方法。通过这个接口，具体的实现可以根据需求来处理图的加载、存储、扩展等操作，通常用于支持基于图的近似最近邻（ANN）算法，例如 HNSW 等图索引方法。
```

```admonish note collapsible=true, title='abstract_index.h'
这段代码定义了一个名为 `AbstractIndex` 的抽象类，用于表示和管理索引（Index）。该类提供了各种操作索引的接口，包括构建索引、搜索、插入、删除、标签管理、优化索引布局等。它通过使用模板方法和类型擦除来支持多种数据类型和标签类型的操作。以下是该类的主要功能和作用概述：

1. **索引构建和加载：**
   - `build`：用于构建索引，从文件或数据源加载数据并建立索引结构。
   - `save`：将构建好的索引保存到文件。
   - `load`：从文件加载已保存的索引。

2. **搜索操作：**
   - `search_with_optimized_layout`：在优化的布局下进行快速搜索。
   - `search_with_tags`：支持带标签的搜索，可以根据标签进行查询，返回相应的数据和距离。
   - `search`：基于查询向量进行搜索，返回与查询向量最相似的 K 个索引。
   - `search_with_filters`：支持带有过滤条件的搜索，可以根据标签等进行过滤。

3. **插入和删除：**
   - `insert_point`：向索引中插入新的数据点，支持带标签和不带标签的插入方式。
   - `lazy_delete`：延迟删除指定标签的数据点，可以逐个删除或批量删除。
   - `get_active_tags`：获取当前活动的标签集合。

4. **索引优化：**
   - `optimize_index_layout`：优化索引的存储布局，提高查询效率。

5. **向量操作：**
   - `get_vector_by_tag`：根据标签检索对应的向量。
   - `set_start_points_at_random`：随机设置起始点，通常用于初始化查询过程。

6. **数据删除合并：**
   - `consolidate_deletes`：合并删除操作，处理数据删除的结果，并返回一个报告。

7. **标签管理：**
   - `set_universal_label`：为所有数据点设置一个统一的标签。

**模板方法和类型擦除：**

- 类中大量使用了模板方法，如 `build`、`search_with_tags`、`insert_point` 等。这些模板方法允许该类支持多种不同的数据类型（如 `data_type`、`tag_type`）和标签类型（如 `label_type`）。
- 使用 `std::any` 实现类型擦除，使得同一个接口可以处理多种不同类型的索引、查询和结果类型，从而增强了类的灵活性和可扩展性。

**私有和保护方法：**

- 除了公开的接口方法外，该类还有一些私有和保护方法（如 `_build`、`_insert_point`、`_lazy_delete` 等）。这些方法是供继承类实现时使用的，继承类需要提供具体的实现。

`AbstractIndex` 类是一个高度抽象的索引管理类，提供了索引的构建、存储、加载、搜索、插入、删除等多种功能。通过模板方法和类型擦除，类能够支持多种数据类型和标签类型，从而提高了灵活性和扩展性。继承自该类的具体实现可以提供不同的数据存储和索引算法，以支持多种不同的场景和需求。
```

```admonish note collapsible=true, title='abstract_scratch.h'
这段代码定义了一个名为 `AbstractScratch` 的抽象类，主要用于存储和管理与查询相关的中间数据。它为处理查询数据提供了一些功能，包括对齐后的查询向量和与产品量化（PQ）相关的中间数据。

1. **成员变量：**
   - `aligned_query_T`：存储对齐后的查询向量，通常用于加速距离计算，尤其是在某些距离度量（如 L2 距离）要求数据对齐时。
   - `pq_scratch`：存储产品量化（PQ）相关的中间数据，可能用于加速高维数据的查询和搜索。

2. **接口方法：**
   - `aligned_query_T()`：返回对齐后的查询向量指针。
   - `pq_scratch()`：返回产品量化（PQ）相关的中间数据指针。

3. **禁止拷贝：**
   - 该类禁止拷贝操作，通过删除拷贝构造函数和拷贝赋值运算符，确保该类的对象不会被复制。

4. **内存管理：**
   - `AbstractScratch` 类并不负责其成员变量的内存管理。内存管理由派生类（如 `InMemQueryScratch` 或 `SSDQueryScratch`）负责，这样可以确保派生类在创建和销毁时正确管理内存。

`AbstractScratch` 类是一个抽象类，提供了一个通用的接口用于处理查询相关的临时数据。它包含了对齐查询向量和产品量化（PQ）中间数据的成员，并为派生类提供了灵活的管理方式。该类的设计旨在让派生类负责内存管理，确保对齐数据和 PQ 数据的有效存储和使用。在查询过程中，派生类可以根据需要对这些数据进行操作，提升查询效率。
```

```admonish note collapsible=true, title='aligned_file_reader.h'
这段代码定义了一个用于文件读取和异步IO操作的类结构，主要用于在不同操作系统（如 Windows 和 Linux）上执行对齐的文件读取操作。它为并发操作和异步IO提供了必要的支持，能够高效地处理大量数据的读取请求。

1. **IOContext 结构：**
   - `IOContext` 结构体用于存储与异步IO操作相关的上下文信息，包括文件句柄、IO完成端口或请求信息等。
   - 在不同的操作系统上，`IOContext` 的实现不同：
     - **Linux：** 使用 `io_context_t` 和 `libaio` 库进行异步IO操作。
     - **Windows：** 使用 `HANDLE` 和 `OVERLAPPED` 结构处理异步IO操作。

2. **AlignedRead 结构：**
   - `AlignedRead` 结构体用于描述对齐的文件读取请求。它包含了：
     - `offset`：读取的起始位置。
     - `len`：读取的字节数。
     - `buf`：存储读取数据的缓冲区。
   - 所有字段要求是 512 字节对齐，这对性能和内存访问优化很重要，尤其在大规模数据读取时。

3. **AlignedFileReader 类：**
   - 该类负责处理文件的读取操作，并支持多线程和异步操作。它提供了如下功能：
     - **线程管理：** 每个线程都有一个特定的 `IOContext`，用于管理和跟踪异步IO请求。可以注册和注销线程。
     - **文件操作：** 支持打开和关闭文件操作。
     - **读取操作：** 支持异步或同步的批量读取操作，能够高效地处理大规模读取请求。

4. **操作系统适配：**
   - 在 **Linux** 上，使用 `libaio` 来执行异步IO操作，使用 `io_context_t` 来管理IO上下文。
   - 在 **Windows** 上，使用 `OVERLAPPED` 结构和 IO 完成端口来执行异步IO。
   - 特别地，在 `USE_BING_INFRA` 宏定义启用时，Windows 环境中将使用专门的 `IDiskPriorityIO` 接口来优化磁盘IO优先级。

5. **异步IO操作：**
   - **Linux：** 支持通过 `io_context_t` 和 `libaio` 执行异步IO操作。
   - **Windows：** 通过 `IDiskPriorityIO` 接口和自定义的 IOContext 管理异步IO操作。
   - `AlignedFileReader` 支持批量对齐请求的处理，并允许通过 `wait` 方法等待所有请求完成。

6. **内存对齐：**
   - 所有读取请求的数据都要求 512 字节对齐，这是为了优化磁盘访问和提升读取效率。`AlignedRead` 结构确保了数据块的对齐。

这段代码设计了一个高效的文件读取系统，支持异步IO操作和多线程环境。通过 `AlignedFileReader` 类和 `IOContext` 结构，程序能够处理大量对齐的读取请求，并在不同操作系统上执行优化的IO操作。它为高性能数据读取场景提供了基础，特别适合在需要处理大规模数据集（如机器学习、数据分析等）时使用。
```

```admonish note collapsible=true, title='ann_exception.h'
这段代码定义了两个异常类，`ANNException` 和 `FileException`，用于在 DiskANN 项目中处理与近似最近邻（ANN）算法和文件操作相关的错误。它们继承自 C++ 标准库中的 `std::runtime_error` 类，并扩展了异常信息的捕捉和报告功能。

1. **ANNException 类：**
   - `ANNException` 继承自 `std::runtime_error`，用于表示与 ANN 相关的异常。
   - **构造函数：**
     - 接受一个错误消息和一个错误码，用于在抛出异常时提供详细的错误信息。
     - 还支持传递额外的函数签名、文件名和行号信息，这有助于调试时定位错误发生的上下文。
   - **成员变量：**
     - `_errorCode`：存储错误代码，帮助区分不同类型的错误。
   - **功能：** 用于在 ANN 算法中抛出错误时提供详细信息，如算法的计算、数据存储或查询操作等问题。

2. **FileException 类：**
   - `FileException` 继承自 `ANNException`，用于处理与文件操作相关的异常。
   - **构造函数：**
     - 接受文件名、系统错误信息（`std::system_error`）以及函数签名、文件名和行号信息。这样可以在抛出异常时提供更多的上下文信息，便于调试。
   - **功能：** 主要用于文件操作失败的情况，例如读取文件失败、文件打开失败等问题。

3. **平台相关自定义：**
   - `#ifndef _WINDOWS` 和 `#define __FUNCSIG__ __PRETTY_FUNCTION__`：这些宏定义用于平台特定的定制。当代码在 Windows 平台上编译时，`__FUNCSIG__` 会被定义为 Windows 环境下的函数签名，而在其他平台上，它会被定义为 `__PRETTY_FUNCTION__`，这是 GCC 和 Clang 编译器中提供的更详细的函数签名信息。

4. **异常类的功能：**
   - 通过继承自 `std::runtime_error`，这两个类能够提供标准的异常功能，如捕获错误信息并在异常处理中传递。
   - 在 `ANNException` 和 `FileException` 中添加了更多的上下文信息（如函数签名、文件名和行号），使得在调试过程中，开发者能够更容易地找到异常的根源。

`ANNException` 和 `FileException` 是自定义的异常类，用于在 DiskANN 项目中处理算法执行和文件操作过程中的错误。它们提供了标准错误消息外，还支持附加更多调试信息（如函数签名、文件名、行号等），有助于开发人员快速定位问题并解决。在平台上，`__FUNCSIG__` 和 `__PRETTY_FUNCTION__` 宏的使用确保了在不同平台下能够获得详细的函数信息。
```

```admonish note collapsible=true, title='any_wrappers.h'
这段代码定义了一些模板结构和类，用于封装不同类型的数据引用，支持在不复制数据的情况下访问它们。通过使用 `std::any` 来存储类型安全的引用，它允许在运行时动态地操作各种不同类型的数据。这些封装类使得数据引用在不同的场景中更加灵活。

1. **AnyReference 结构：**
   - 这是一个基础结构，用于存储和访问对任意类型数据的引用。
   - **构造函数：**
     - 通过模板构造函数，`AnyReference` 可以接受任何类型的数据引用（`Ty &`），并将其保存在 `_data` 成员中。
   - **get() 方法：**
     - `get` 方法返回对存储数据的引用。它使用 `std::any_cast<Ty *>` 将 `std::any` 中的数据转换为特定类型的指针，并解引用后返回。这提供了对存储数据的访问。
   - **注意：** 该结构本身不进行内存管理，调用者需要确保传入的数据在使用期间保持有效。

2. **AnyRobinSet 结构：**
   - `AnyRobinSet` 继承自 `AnyReference`，用于存储 `tsl::robin_set` 的引用。
   - 通过模板构造函数，`AnyRobinSet` 可以接受一个常量引用或非常量引用的 `tsl::robin_set` 对象，并将其传递给 `AnyReference` 构造函数。
   - `AnyRobinSet` 使得可以存储和访问 `robin_set` 类型的数据，并通过 `AnyReference` 提供对其元素的访问。

3. **AnyVector 结构：**
   - `AnyVector` 继承自 `AnyReference`，用于存储 `std::vector` 的引用。
   - 类似于 `AnyRobinSet`，`AnyVector` 也通过模板构造函数接收常量或非常量引用的 `std::vector` 对象，并将其传递给 `AnyReference` 的构造函数。
   - `AnyVector` 允许封装 `std::vector`，使得可以通过 `AnyReference` 来访问存储在 `vector` 中的元素。

- **`AnyReference`** 是一个通用的数据包装器，可以封装对任意类型数据的引用，并通过 `get` 方法访问该数据。它是 `AnyRobinSet` 和 `AnyVector` 的基础类。
- **`AnyRobinSet`** 和 **`AnyVector`** 分别是 `AnyReference` 的特化版本，用于存储和操作 `tsl::robin_set` 和 `std::vector` 类型的数据。
- 这些类通过 `std::any` 提供了类型安全的封装，允许在运行时灵活地存储和访问不同类型的数据引用。

这些类的设计使得在程序中可以动态地操作不同类型的数据容器，而无需担心数据的复制或类型不匹配。它们非常适合在泛化的上下文中使用，尤其是当你需要处理不同类型的容器时。
```

```admonish note collapsible=true, title='boost_dynamic_bitset_fwd.h'
这段代码是一个简单的头文件，它涉及到 `boost::dynamic_bitset` 类模板的声明。

1. **防止重复定义：**
   - `#ifndef BOOST_DYNAMIC_BITSET_FWD_HPP` 和 `#endif` 宏的作用是防止重复定义。在第一次包含头文件时，如果 `BOOST_DYNAMIC_BITSET_FWD_HPP` 没有被定义，它就会继续处理文件内容；如果已经定义过，文件中的内容就会被忽略。
   - 这种方法通常用于前向声明，以避免在多个地方重复包含同一部分代码。

2. **`boost::dynamic_bitset` 类模板：**
   - 该模板类是 Boost 库中的一个数据结构，用于处理动态大小的位集（bitset）。与标准库中的固定大小的 `std::bitset` 不同，`boost::dynamic_bitset` 可以在运行时动态增加或减少大小。
   - 该类模板接收两个模板参数：
     - `Block`：用于存储位集的单个块的类型，默认为 `unsigned long`。位集通常以块为单位来管理，`Block` 类型决定了每个块的大小。
     - `Allocator`：用于分配内存的分配器类型，默认为 `std::allocator<Block>`，它是标准的内存分配器。

3. **目的：**
   - 该代码的目的是声明 `boost::dynamic_bitset` 类模板的前向声明，而没有提供其完整实现。完整的实现通常在另一个头文件中。这样做的好处是，如果其他代码只需要知道 `dynamic_bitset` 的存在而不需要其完整实现时，可以通过前向声明来减少编译依赖，从而加快编译速度。

这段代码是 `boost::dynamic_bitset` 类模板的前向声明，它允许程序在使用 `boost::dynamic_bitset` 时仅依赖其声明，而无需包含完整的实现。这对于加速编译过程和减少依赖关系是非常有用的。`dynamic_bitset` 类是一个动态大小的位集容器，提供了灵活的位操作功能。
```

```admonish note collapsible=true, title='cached_io.h'

```

```admonish note collapsible=true, title=''

```

```admonish note collapsible=true, title=''

```

```admonish note collapsible=true, title=''

```

```admonish note collapsible=true, title=''

```

```admonish note collapsible=true, title=''

```

```admonish note collapsible=true, title=''

```

```admonish note collapsible=true, title=''

```