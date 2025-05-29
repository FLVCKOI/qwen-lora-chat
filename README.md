Qwen-1.8B LoRA智能助手
Python PyTorch HF
🌟 核心功能
基于QLoRA的4-bit量化微调（节省70%显存）
支持动态批处理与梯度检查点
集成REST API与WebSocket双协议接口
自适应温度调节（0.1-2.0）与重复惩罚机制
🚀 快速启动
环境要求
NVIDIA GPU ≥ RTX 3090 (24GB VRAM)
CUDA 11.8 • Ubuntu 20.04+
三步部署
# 1. 克隆仓库 
git clone https://github.com/yourusername/qwen-lora-chat  
 
# 2. 安装依赖（建议使用虚拟环境）
pip install -r requirements.txt  
 
# 3. 启动服务 
python main.py  --port 8000 --quantize
⚙️ 高级配置
Docker部署
docker build -t qwen-chat .
docker run -d --gpus all -p 8000:8000 \
  -v ./models:/app/models \
  qwen-chat --max_tokens 2048
API调用示例
import requests 
 
payload = {
    "prompt": "用Python实现快速排序",
    "temperature": 0.9,
    "max_tokens": 500 
}
response = requests.post("http://localhost:8000/generate",  json=payload)
print(response.json()["response"])
📁 项目结构
├── configs/            # 配置文件 
│   ├── train.yaml       # 训练参数配置 
│   └── inference.yaml   # 推理参数配置 
├── data/               # 数据处理模块 
├── docker/             # 容器化配置 
├── models/             # 模型文件（自动下载）
├── requirements.txt     # Python依赖 
└── README.html          # 本说明文件
⚠️ 重要提示
• 首次运行会自动下载约8GB的预训练模型文件

• 建议使用Git LFS管理大文件：

git lfs install 
git lfs track "*.safetensors"
git add .gitattributes
