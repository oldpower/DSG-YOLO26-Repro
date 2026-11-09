# DSG-YOLO26 结构级复现

基于 Ultralytics 的 **DSG-YOLO26 未训练网络搭建**，对应论文：

> Si Chen, Shuxian Liu. *DSG-YOLO26: Detail-Preserving and Scale-Gated YOLO for UAV small object detection*. Signal Processing: Image Communication, 149 (2026), 117677.
> DOI: [10.1016/j.image.2026.117677](https://doi.org/10.1016/j.image.2026.117677)

## 复现状态与边界

**本项目是依据公开摘要完成的结构级、工程化复现，不是作者官方源码，也不是逐层精确复刻。**

截至 2026-11-09，论文正文未开放；[同名 GitHub 仓库](https://github.com/ChenSi-0819/DSG-YOLO26)及其历史提交只有一篇标注 “Coming soon.” 的 README，没有 YAML、模型代码、结构图或权重。因此，本项目严格区分以下内容：

| 项目 | 公开资料能够确认的内容 | 本项目的工程化补全 |
| --- | --- | --- |
| 基线 | YOLO26s | 使用 Ultralytics YOLO26-P2 拓扑，保留 P2/P3/P4/P5 四个输出；其他规模是扩展项 |
| DPSE | 在 P2 分支使用轻量残差深度卷积 | 3×3 depthwise + 1×1 pointwise + 残差；重复次数随模型深度缩放 |
| SGCF | 上采样 P3 语义在进入 P2 前进行尺度门控 | P3/P2 分别投影，逐通道 softmax 跨尺度加权，再做卷积精炼 |
| SSR | 训练时重加权定位监督，推理零额外开销 | 提供独立的面积幂次权重工具，默认 `area^-0.5`；**不接入推理图，也未冒充论文公式** |
| 检测头 | 基于 YOLO26 | 原生 `Detect`，`reg_max=1`，创建 one-to-many/one-to-one 双分支 |

公开摘要声称 VisDrone2019 上达到 0.535 mAP50、0.340 mAP50-95、约 9.6M 参数；本项目**没有训练或复现这些指标**。当前实现的 DSG-YOLO26s 实测为 **9,794,352 参数**，不能把论文指标或精确参数量视为已验证结果。

## 环境

严格使用已有环境，不安装、升级或修改任何依赖：

```bash
conda activate env01
```

本次实际验证环境：

- Python 3.11.18
- Ultralytics 8.4.198
- PyTorch 2.15.0+cpu

自定义解析器和模块均在本仓库内，不修改 `site-packages`，也不 monkeypatch Ultralytics 的全局模型注册表。

## 快速开始

### 1. 命令行构建并检查前向形状

```bash
# 默认 scale=s；64 仅用于快速形状测试，不代表训练分辨率
python build_model.py --scale s --nc 80 --imgsz 64

# 仅构建，不执行额外的随机输入前向
python build_model.py --scale s --skip-forward --quiet
```

默认 `s` 规模的实测输出：

```text
Model: DSG-YOLO26s
Parameters: 9,794,352 (trainable: 9,794,352)
Detection levels: 4; strides: [4.0, 8.0, 16.0, 32.0]
YOLO26 head: one2one_branch=True, end2end_active=False, reg_max=1, nc=80
Input: (1, 3, 64, 64)
P2/4 output: (1, 64, 16, 16)
P3/8 output: (1, 128, 8, 8)
P4/16 output: (1, 256, 4, 4)
P5/32 output: (1, 512, 2, 2)
```

`end2end_active=False` 与当前 Ultralytics 原生 YOLO26 YAML 的初始行为一致：配置会创建 one-to-one 分支，但初始选中状态不是端到端推理。需要检查 NMS-free 输出时，可显式设置 `model.model[-1].end2end = True`。

### 2. Python API

```python
from dsg_yolo26 import DSGYOLO26, build_dsg_yolo26

# 直接获得 Ultralytics DetectionModel 子类
network = build_dsg_yolo26(scale="s", nc=80, ch=3, verbose=False)
print(network.stride)  # tensor([4., 8., 16., 32.])
network.info()

# 或获得带 Ultralytics 元数据的高级 YOLO 封装
model = DSGYOLO26(scale="s", nc=80, ch=3, verbose=False)
print(type(model.model).__name__)  # DSGDetectionModel
```

**不要直接调用 `YOLO("configs/dsg-yolo26.yaml")`**：原生 Ultralytics 解析器不认识本仓库的 `DPSE`/`SGCF`，请使用上述入口。

### 3. SSR 训练辅助工具

```python
import torch
from dsg_yolo26 import ScaleSensitiveRegression

ssr = ScaleSensitiveRegression(power=-0.5, box_format="xyxy")
boxes = torch.tensor([[0, 0, 8, 8], [0, 0, 32, 32]], dtype=torch.float32)
weights = ssr(boxes)  # 有效框权重均值归一为 1；小框权重大于大框
```

该工具只生成权重。如何将它乘入 CIoU/DIoU/其他定位损失，需要自定义训练器；本次 PRD 明确不要求训练，因此未实现训练器，也未将 SSR 加入任何推理路径。

## 网络数据流

```text
YOLO26 backbone
  ├─ P2 shallow details ───────────────┐
  ├─ P4/P5 → top-down P3 semantics     │
  │                         ↓ Upsample │
  │                              SGCF ←┘
  │                               ↓
  │                             DPSE → P2 detection
  │                               ↓
  │                         bottom-up PAN
  │                         ├─ P3 detection
  │                         ├─ P4 detection
  │                         └─ P5 detection
  └─ 原生 YOLO26 Detect（reg_max=1，双分支）

SSR：仅训练监督权重，不在上述推理图中。
```

配置中的 P2/P3/P4/P5 四头、stride、通道数和逐层连接属于明确的工程化假设；公开摘要并没有给出这些细节。

## 测试

不需要 pytest，直接使用 Python 标准库：

```bash
python -m unittest discover -s tests -v
```

当前共 **9 项测试**，覆盖：

- DPSE 分辨率保持、深度卷积分组和梯度传播；
- SGCF 初始 50/50 尺度门控、逐通道归一化、双输入梯度及形状校验；
- SSR 无效框处理、有效框权重归一化及零可训练参数；
- 自定义模块位置、四尺度 stride、原生 YOLO26 检测头；
- 64×64 与 96×96 输入的训练模式前向形状；
- 显式启用 one-to-one 后的端到端推理输出；
- 高级封装元数据及非法规模参数校验。

## 文件结构

```text
configs/dsg-yolo26.yaml       # 网络拓扑与 n/s/m/l/x 缩放
dsg_yolo26/modules.py         # DPSE、SGCF
dsg_yolo26/ssr.py             # 独立、训练专用 SSR 权重工具
dsg_yolo26/parser.py          # 限定范围的 Ultralytics YAML 解析器
dsg_yolo26/model.py           # DetectionModel 与高级 YOLO 封装
build_model.py                # 构建和随机输入形状检查入口
tests/test_dsg_yolo26.py      # 标准库单元测试
```

## 公开依据

- [论文 DOI / Crossref 元数据](https://api.crossref.org/works/10.1016/j.image.2026.117677)
- [高度疑似作者仓库的公开 README](https://raw.githubusercontent.com/ChenSi-0819/DSG-YOLO26/main/README.md)
- [公开仓库当前文件树：仅 README](https://api.github.com/repos/ChenSi-0819/DSG-YOLO26/git/trees/main?recursive=1)
- [Ultralytics YOLO26 文档](https://docs.ultralytics.com/models/yolo26)

若后续取得作者正文或正式源码，应优先替换本项目的 DPSE/SGCF 公式、P2 连接假设和 SSR 权重函数，再开展训练与指标对照。
