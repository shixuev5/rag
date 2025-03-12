# RAG 文档检索系统

基于 BGE-M3 和 Milvus 的文档检索系统，支持向量检索、BM25检索和混合检索策略。

## 功能特点

- 文档向量化：使用 BGE-M3 模型进行文档向量化
- 混合检索：结合向量检索和 BM25 检索
- 重排序：使用 BGE-Reranker-V2-M3 模型对检索结果进行重排序
- REST API：提供标准的 REST API 接口
- 元数据过滤：支持基于文档元数据的过滤
- 批量处理：支持批量文档处理
- 可扩展性：模块化设计，易于扩展
- 容器化部署：支持 Docker 容器化部署

## 系统要求

- Docker 和 Docker Compose
- 或者：
  - Python 3.8+
  - Milvus 2.3+
  - BGE-M3 模型服务
  - BGE-Reranker-V2-M3 模型服务

## 部署方式

### 本地部署

1. 克隆代码仓库：
```bash
git clone [repository_url]
cd rag
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境：
   - 确保 Milvus 服务已启动
   - 确保模型服务已启动并可访问
   - 根据需要修改 `src/config/settings.py` 中的配置

4. 启动服务：
```bash
python src/main_api.py
```

## API 使用示例

1. 文档向量化：
```bash
curl -X POST "http://localhost:8000/vectorize" \
     -H "Content-Type: application/json" \
     -d '{
         "texts": ["示例文本1", "示例文本2"],
         "batch_size": 32
     }'
```

2. 文档检索：
```bash
curl -X POST "http://localhost:8000/retrieval" \
     -H "Content-Type: application/json" \
     -d '{
         "knowledge_id": "test-knowledge",
         "query": "示例查询",
         "retrieval_setting": {
             "top_k": 5,
             "score_threshold": 0.5
         }
     }'
```

## 配置说明

### 环境变量

容器化部署时可以通过环境变量配置：

- `MILVUS_HOST`: Milvus 服务地址
- `MILVUS_PORT`: Milvus 服务端口
- `MODEL_SERVICE_HOST`: 模型服务地址
- `MODEL_SERVICE_PORT`: 模型服务端口

### 配置文件

主要配置项（`src/config/settings.py`）：

```python
# Milvus配置
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530

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
    "WORKERS": 4
}
```

## 开发说明

项目结构：
```
src/
├── main_api.py          # API服务启动脚本
├── main.py             # 命令行工具脚本
├── api/
│   ├── service.py     # API服务实现
│   └── schemas.py     # API数据模型
├── config/
│   └── settings.py    # 配置文件
├── models/
│   └── client.py      # 模型服务客户端
├── retrievers/
│   ├── hybrid_retriever.py  # 混合检索实现
│   └── base.py       # 检索基类
└── rankers/
    └── reranker.py   # 重排序实现
```

## 注意事项

1. 向量维度：BGE-M3 模型输出的向量维度为 1024
2. 模型服务：需要确保 BGE-M3 和 BGE-Reranker-V2-M3 模型服务正常运行
3. 内存使用：注意监控 Milvus 的内存使用情况，特别是在处理大量文档时
4. 容器化部署：
   - 首次启动时，Milvus 可能需要一些时间初始化
   - 确保为容器分配足够的内存
   - 在生产环境中，建议为 Milvus 配置持久化存储

## 许可证

[添加许可证信息] 