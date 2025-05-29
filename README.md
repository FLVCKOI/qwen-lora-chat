
````markdown
# 🔮 Qwen-1.8B LoRA 智能助手

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)

---

## 🌟 核心功能

- 基于 QLoRA 的 4-bit 量化微调（节省约 70% 显存）
- 支持动态批处理与梯度检查点（Gradient Checkpointing）
- 集成 REST API 与 WebSocket 双协议接口
- 支持自适应温度调节（0.1~2.0）与重复惩罚机制

---

## 🚀 快速启动

### 环境要求

```text
NVIDIA GPU ≥ RTX 3090 (24GB VRAM)
CUDA 11.8 • Ubuntu 20.04+
````

### 三步部署

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/qwen-lora-chat

# 2. 安装依赖（建议使用虚拟环境）
pip install -r requirements.txt

# 3. 启动服务
python main.py --port 8000 --quantize
```

---

## ⚙️ 高级配置

### Docker 部署

```bash
docker build -t qwen-chat .
docker run -d --gpus all -p 8000:8000 \
  -v ./models:/app/models \
  qwen-chat --max_tokens 2048
```

### API 调用示例

```python
import requests

payload = {
    "prompt": "用Python实现快速排序",
    "temperature": 0.9,
    "max_tokens": 500
}
response = requests.post("http://localhost:8000/generate", json=payload)
print(response.json()["response"])
```

---

## 📁 项目结构

```
├── configs/            # 配置文件
│   ├── train.yaml       # 训练参数配置
│   └── inference.yaml   # 推理参数配置
├── data/               # 数据处理模块
├── docker/             # 容器化配置
├── models/             # 模型文件（自动下载）
├── requirements.txt    # Python 依赖
└── README.md           # 本说明文件
```

---

## ⚠️ 重要提示

* ⚠️ **首次运行会自动下载约 8GB 的预训练模型文件**
* 建议使用 [Git LFS](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/about-git-large-file-storage) 管理大文件：

```bash
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
```

