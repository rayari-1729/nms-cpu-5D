# 📦 cpu-rotated-nms

> **Lightweight CPU-based 3D Non-Maximum Suppression (NMS)** with rotation support  
> Drop-in replacement for `nms_gpu` — works with 5D bounding boxes: `[x1, y1, x2, y2, ry]`  
> ⚡️ No CUDA required | 🚀 On-device ready | 🧠 PyTorch compatible

---

## 🔧 Features

- ✅ CPU-compatible (no `iou3d_cuda` dependency)
- ✅ Keeps `ry` (rotation) for accurate 3D box filtering
- ✅ Supports `pre_maxsize` and `post_max_size`
- ✅ Easy to plug into existing 3D detection models

---

## 📐 Input Format

- `boxes`: `torch.Tensor` of shape `[N, 5]`  
  → Format: `[x1, y1, x2, y2, ry]`
- `scores`: `torch.Tensor` of shape `[N]`
- `iou_threshold`: float (e.g. `0.4`)

---

## 🧪 Usage

```python
from nms_cpu import nms_cpu

keep_indices = nms_cpu(
    boxes,          # torch.Tensor [N, 5]
    scores,         # torch.Tensor [N]
    iou_threshold=0.5,
    pre_maxsize=1000,
    post_max_size=100
)
```
---

## 📥 Installation

🔹 Option 1: From GitHub
```bash
`pip install git+https://github.com/rayari-1729/nms-cpu-5D.git`
```

🔹 Option 2: From Local
```bash
git clone https://github.com/rayari-1729/nms-cpu-5D.git
cd nms_cpu
pip install -e .
```

