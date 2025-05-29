# Qwen-1.8B LoRA智能助手 🚀 
 
![Python](https://img.shields.io/badge/Python-3.10%2B-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red) 
![HF](https://img.shields.io/badge/HuggingFace-Transformers-orange) 
 
## 🌟 核心功能 
- **高效微调**：采用4-bit QLoRA量化技术，显存占用降低70%
- **智能生成**：支持动态温度调节（0.1-2.0）与2048 tokens长文本生成 
- **多协议接口**：集成REST API + WebSocket双通信协议 
- **生产就绪**：提供Docker容器化部署方案 
 
## 🚀 快速部署 
### 基础环境要求 
```bash 
# 最低硬件配置 
GPU：NVIDIA RTX 3090 (24GB VRAM)
CUDA：11.8+
OS：Ubuntu 20.04 LTS
三步启动方案
bash
复制
git clone https://github.com/yourrepo/qwen-lora-chat  
pip install -r requirements.txt  
python main.py  --port 8000 --quantize 
⚙️ 高级配置
Docker部署
dockerfile
复制
docker build -t qwen-chat .
docker run -d --gpus all -p 8000:8000 \
  -v ./models:/app/models \
  qwen-chat --max_tokens 2048 
API调用示例
python
复制
import requests 
 
payload = {
    "prompt": "用Python实现快速排序",
    "temperature": 0.9,
    "max_tokens": 500 
}
response = requests.post("http://localhost:8000/generate",  json=payload)
print(response.json()["response"]) 
📁 项目结构
.
├── configs/            # 训练/推理配置 
├── data/               # 数据处理模块 
├── docker/             # 容器化配置 
├── models/             # 模型存储目录（自动下载）
└── README.md            # 本说明文档 
⚠️ 注意事项
模型下载：首次运行会自动下载约8GB的预训练模型
大文件管理：建议使用Git LFS：
bash
复制
git lfs install 
git lfs track "*.safetensors"
git add .gitattributes 
