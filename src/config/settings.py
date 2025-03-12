from pathlib import Path

# 路径配置
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

# Milvus配置
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "markdown_docs"
VECTOR_DIM = 1024  # BGE-M3 模型的向量维度

# 文档处理配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 检索配置
INITIAL_RETRIEVAL_SIZE = 10  # 初始检索数量
FINAL_RETRIEVAL_SIZE = 5    # 重排序后返回数量
DEFAULT_VECTOR_WEIGHT = 0.7  # 向量检索权重

# 模型配置
EMBEDDING_MODEL_NAME = "bge-m3"
RERANK_MODEL_NAME = "bge-reranker-v2-m3"

# 模型服务配置
MODEL_SERVICE = {
    "HOST": "localhost",
    "PORT": 8080,
    "EMBEDDING_PATH": "/v1/embeddings",
    "RERANK_PATH": "/v1/rerank",
    "TIMEOUT": 30,
}

# API服务配置
API_SETTINGS = {
    "HOST": "0.0.0.0",
    "PORT": 8000,
    "WORKERS": 4,
    "DEBUG": True
}

# 批处理设置
VECTORIZE_BATCH_SIZE = 32

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