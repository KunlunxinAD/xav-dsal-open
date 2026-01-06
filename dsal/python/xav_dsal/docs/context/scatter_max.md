## scatter_max
### 接口原型
```python
xav_dsal.scatter_max(Tensor src, Tensor indices, int dim=0， Tensor out=None, int dim_size=None, Tensor offset=None, int offset_reduce=None) -> (Tensor, Tensor)
```
### 功能描述
在第0维上，将输入张量`src`中的元素按照`indices`中的索引进行分散，然后在第0维上取最大值，返回最大值和对应的索引。对于1维张量，公式如下：
$$out_i = max(out_i, max_j(src_j))$$
$$argmax_i = argmax_j(src_j)$$
这里，$i = indices_j$。
### 参数说明
- `src (Tensor)`：源张量 (Tensor)，数据类型为`float32`。
- `indices (Tensor)`：索引张量 (Tensor)，数据类型为`long`。
- `out (Tensor)`：被更新张量 (Tensor)，数据类型为`float32`，可选入参，默认为`None`，输入`out`不为`None`时，`out`中的元素参与求和的计算。
- `dim (int)`：指定的维度，表示按照哪个维度进行分组计算，数据类型为`int32`，可选入参，默认取值为`0`。
- `dim_size (int)`：输出张量在`dim`维的长度，数据类型为`int32`，可选入参，默认为`None`，该参数仅在输入`out`为`None`时生效。
- `offset (Tensor)`：维度必须是1。
- `offset_reduce (int)`: 只支持SUM（0）或者MUL（2）。
### 返回值
- `out (Tensor)`：求最大值的张量 (Tensor)，数据类型为`float32`。
- `argmax (Tensor)`: 最大值对应的索引，类型为`int32`.
### 算子约束
- src.dim == index.dim
- 在除了 scatter 的维度上，src.size == out.size
- 若index.dim 为1，index的长度<=src在dim维的长度；若index.dim不为1，index每一维长度<=src对应维的长度
- `dim`取值不能超过`indices`的维度。
- `dim_size`的取值必须为非负的有效长度值。
### 支持的型号
- P800
### 调用示例
```python
import torch
from xav_dsal import scatter_max
src = torch.tensor([[2, 0, 1, 3, 1, 0, 0, 4], [0, 2, 1, 3, 0, 3, 4, 2], [1, 2, 3, 4, 4, 3, 2, 1]], dtype=torch.float32).to("cuda")
indices = torch.tensor([0, 2, 0], dtype=torch.int64).to("cuda")
dim = 0
out, argmax = scatter_max(src, indices, dim, None)
```
