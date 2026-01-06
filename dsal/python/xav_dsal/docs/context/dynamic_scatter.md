## dynamic_scatter
### 接口原型
```python
xav_dsal.dynamic_scatter(Tensor feats, Tensor coors, str reduce_type = 'max') -> Tensor[]
```
### 功能描述
将点云特征点在对应体素中进行特征压缩。
### 参数说明
- `feats(Tensor)`：点云特征张量[N, C]，仅支持两维，数据类型为`float32`，特征向量`C`长度上限为2048。
- `coors(Tensor)`：体素坐标映射张量[N, 3]，仅支持两维，数据类型为`int32/float32`，此处以x, y, z指代体素三维坐标，其取值范围为`0 <= x, y < 2048`,  `0 <= z < 256`。
- `reduce_type(str)`：压缩类型。可选值为`'max'`, `'mean'`, `'sum'`。默认值为`'max'`
### 返回值
- `voxel_feats(Tensor)`：压缩后的体素特征张量，仅支持两维，数据类型为`float32`。
- `voxel_coors(Tensor)`：去重后的体素坐标，仅支持两维，数据类型为`int32`。
### 支持的型号
- p800
### 调用示例
```python
import torch
from xav_dsal import dynamic_scatter

feats = torch.randn(30, 4).to("cuda")
coors = torch.randn((30, 3), dtype=torch.float32).to("cuda")
reduce = "mean"
voxel_feats_cuda, voxel_coors_cuda, point2voxel_map_cuda, voxel_points_count_cuda = dynamic_scatter(feats, coors, reduce)

```