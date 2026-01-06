## scatter_add
### 接口原型
```python
xav_dsal.scatter_add(Tensor src, Tensor indices, int dim=0， Tensor out=None, int dim_size=None, Tensor offset=None, int offset_reduce=None) -> Tensor
```
### 功能描述
将输入张量`src`中的元素按照`indices`中的索引在指定的`dim`维进行分组，并对每组进行求和，求和后的结果放在`out`中。
### 参数说明
- `src (Tensor)`：源张量 (Tensor)，数据类型为`float32`。
- `indices (Tensor)`：索引张量 (Tensor)，数据类型为`int64`。
- `out (Tensor)`：被更新张量 (Tensor)，数据类型为`float32`，可选入参，默认为`None`，输入`out`不为`None`时，`out`中的元素参与求和的计算。
- `dim (int)`：指定的维度，表示按照哪个维度进行分组求和计算，数据类型为`int32`，可选入参，默认取值为`0`。
- `dim_size (int)`：输出张量在`dim`维的长度，数据类型为`int32`，可选入参，默认为`None`，该参数仅在输入`out`为`None`时生效。
- `offset (Tensor)`：维度必须是1。
- `offset_reduce (int)`: 只支持SUM（0）或者MUL（2）。
### 返回值
- `out (Tensor)`：求和后的张量 (Tensor)，数据类型为`float32`。
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
from xav_dsal import scatter_add
src = torch.randn(4, 5, 6).to(torch.float)
indices = torch.randint(5, (4, 5)).to(torch.int64)
dim = 0
out = scatter_add(src.to("cuda"), indices.to("cuda"), dim, None)
```
