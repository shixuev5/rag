import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Milvus配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
VECTOR_DIM = 1024  # BGE-M3 模型的向量维度

# 文档处理配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# 模型配置
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "bge-m3")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "bge-reranker-v2-m3")

# 模型服务配置
MODEL_SERVICE = {
    "BASE_URL": os.getenv("MODEL_SERVICE_BASE_URL", "http://localhost:5000"),
    "EMBEDDING_PATH": os.getenv("MODEL_SERVICE_EMBEDDING_PATH", "/v1/embeddings"),
    "RERANK_PATH": os.getenv("MODEL_SERVICE_RERANK_PATH", "/v1/rerank"),
    "CHAT_PATH": os.getenv("MODEL_SERVICE_CHAT_PATH", "/v1/chat/completions"),
    "TIMEOUT": int(os.getenv("MODEL_SERVICE_TIMEOUT", "30")),
}

# API服务配置
API_SETTINGS = {
    "HOST": os.getenv("API_HOST", "0.0.0.0"),
    "PORT": int(os.getenv("API_PORT", "8000")),
    "WORKERS": int(os.getenv("API_WORKERS", "4")),
    "DEBUG": os.getenv("API_DEBUG", "true").lower() == "true"
}

# 错误码定义
ERROR_CODES = {
    # 资源相关错误 (2000-2999)
    2001: "The knowledge does not exist",
    
    # 服务相关错误 (3000-3999)
    3001: "Vectorization failed",
    3002: "Retrieval failed",
    
    # 参数相关错误 (4000-4999)
    4001: "Invalid parameter value",
} 