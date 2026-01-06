## boxes_iou3d
### 接口原型
```python
xav_dsal.boxes_iou3d(Tensor boxes_a, Tensor boxes_b) -> Tensor
```
### 功能描述
计算BEV视角下两个边界框的3D IoU（intersection over union）。
### 参数说明
- `boxes_a (Tensor)`：第一组bounding boxes，数据类型为`float32`。shape为`[M, 7]`。其中`7`分别代表`x_center, y_center, z_center, dx, dy, dz, angle`, `x_center, y_center, z_center`代表box的中心点坐标，`dx, dy, dz`代表box的长宽高，`angle`代表box的弧度制旋转角。
- `boxes_b (Tensor)`：第二组bounding boxes，数据类型为`float32`。shape为`[N, 7]`。其中`7`分别代表`x_center, y_center, z_center, dx, dy, dz, angle`, `x_center, y_center, z_center`代表box的中心点坐标，`dx, dy, dz`代表box的长宽高，`angle`代表box的弧度制旋转角。
### 返回值
- `ious (Tensor)`：包含两组bounding boxes的IoU的张量，数据类型为`float32`。shape为`[M, N]`。
### 约束说明
- `angle`的值在`[-pi, pi]`之间。
### 支持的型号
- p800
### 调用示例
```python
import torch
from xav_dsal import boxes_iou3d
boxes_a = torch.tensor([[1.0, 1.0, 1.0, 3.0, 4.0, 1.0, 0.5]], dtype=torch.float32).to("cuda")
boxes_b = torch.tensor([[0.0, 2.0, 1.0, 2.0, 5.0, 1.0, 0.3]], dtype=torch.float32).to("cuda")
ious = boxes_iou3d(boxes_a, boxes_b)
```