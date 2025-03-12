FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY requirements.txt .
COPY .env.example .env
COPY src/ src/

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV PYTHONPATH=/app
ENV MILVUS_HOST=milvus
ENV MODEL_SERVICE_HOST=model-service

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "src/main_api.py"] 