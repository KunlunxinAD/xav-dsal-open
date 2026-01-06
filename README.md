# xav-dsal

## 简介
xav-dsal是基于昆仑芯XPU平台开发的适用于自动驾驶场景，具身智能VLA及世界模型的算子加速库，提供一系列高性能算子接口。

## 版本说明
DrivingSDK算子支持的CPU架构，Python，PyTorch和torch_npu版本对应关系如下：

| Gitcode分支 |  CPU架构 |  支持的Python版本 | 支持的PyTorch版本 |
|-----------|-----------|-------------------|-------------------|
| master    | x86|Python3.10.x|2.5.0|

## 环境部署
提供`iregistry.baidu-int.com/kunlunxin-self-driving/xav:v1.4.0`镜像进行安装
### 源码安装
1. 克隆原始仓库
```
git clone https://github.com/KunlunxinAD/xav-dsal-open.git
```
2.编译whl包
```
cd dsal && make dist
```
生成的whl包在`dsal/output`中。

## 算子清单
请详见[算子清单](./dsal/python/xav_dsal/docs/README.md)
