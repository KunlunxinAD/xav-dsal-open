## roipoint_pool3d
### 接口原型
```python
xav_dsal.roipoint_pool3d(int num_sampled_points, Tensor points, Tensor point_features, Tensor boxes3d) -> (Tensor pooled_features, Tensor pooled_empty_flag)
```
### 功能描述
对每个3D方案的几何特定特征进行编码。
### 参数说明
- `num_sampled_points(int)`：特征点的数量，正整数。
- `points(Tensor)`：点张量，数据类型为`float32, float16`。shape 为`[B, N, 3]`。`3`分别代表`x, y, z`。
- `point_features(Tensor)`：点特征张量，数据类型为`float32, float16`。shape 为`[B, N, C]`。`C`分别代表`x, y, z`。
- `boxes3d(Tensor)`：框张量，数据类型为`float32, float16`。shape 为`[B, M, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
### 返回值
- `pooled_features(Tensor)`：点在框内的特征张量，数据类型为`float32, float16`。shape 为`[B, M, num, 3+C]`。
- `pooled_empty_flag(Tensor)`：所有点不在框内的空标记张量，数据类型为`int32`。shape 为`[B, M]`。
### 约束说明
- `points`, `point_features`, `boxes3d`, `pooled_features` 所指向元素的数据类型应保持一致。
- `num_sampled_points`必须小于等于`N`。
### 支持的型号
- p800
### 调用示例
```python
import torch
from xav_dsal import roipoint_pool3d
num_sampled_points = 1
points = torch.tensor([[[1, 2, 3]]], dtype=torch.float).to("cuda")
point_features = points.clone()
boxes3d = torch.tensor([[[1, 2, 3, 4, 5, 6, 1]]], dtype=torch.float).to("cuda")
pooled_features, pooled_empty_flag = roipoint_pool3d(points, point_features, boxes3d, num_sampled_points)
```