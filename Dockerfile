# 使用 NVIDIA NGC 官方 PyTorch 镜像（支持 CUDA）
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app


RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list \
 && sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list \
 && apt update && apt install -y git python3-pip && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt \
    && pip install --no-cache-dir tiktoken

# 拷贝项目文件
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]