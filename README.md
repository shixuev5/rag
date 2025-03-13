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
- 灵活配置：支持通过环境变量和配置文件进行配置

## 系统要求

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
```bash
cp .env.example .env
# 编辑 .env 文件，根据需要修改配置
```

4. 启动服务：
```bash
python src/main_api.py
```

### Docker 部署

1. 构建镜像：
```bash
docker build -t rag-api .
```

2. 运行容器：
```bash
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  -e MILVUS_HOST=your-milvus-host \
  -e MODEL_SERVICE_HOST=your-model-service-host \
  rag-api
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

系统支持通过环境变量或 `.env` 文件进行配置。主要配置项包括：

#### Milvus配置
- `MILVUS_HOST`: Milvus 服务地址（默认：localhost）
- `MILVUS_PORT`: Milvus 服务端口（默认：19530）
- `COLLECTION_NAME`: 集合名称（默认：markdown_docs）

#### 文档处理配置
- `CHUNK_SIZE`: 文档分块大小（默认：500）
- `CHUNK_OVERLAP`: 分块重叠大小（默认：50）

#### 检索配置
- `INITIAL_RETRIEVAL_SIZE`: 初始检索数量（默认：10）
- `FINAL_RETRIEVAL_SIZE`: 重排序后返回数量（默认：5）
- `DEFAULT_VECTOR_WEIGHT`: 向量检索权重（默认：0.7）

#### 模型配置
- `EMBEDDING_MODEL_NAME`: 向量化模型名称（默认：bge-m3）
- `RERANK_MODEL_NAME`: 重排序模型名称（默认：bge-reranker-v2-m3）

#### 模型服务配置
- `MODEL_SERVICE_HOST`: 模型服务地址（默认：localhost）
- `MODEL_SERVICE_PORT`: 模型服务端口（默认：8080）
- `MODEL_SERVICE_EMBEDDING_PATH`: 向量化接口路径（默认：/v1/embeddings）
- `MODEL_SERVICE_RERANK_PATH`: 重排序接口路径（默认：/v1/rerank）
- `MODEL_SERVICE_TIMEOUT`: 请求超时时间（默认：30秒）

#### API服务配置
- `API_HOST`: API服务地址（默认：0.0.0.0）
- `API_PORT`: API服务端口（默认：8000）
- `API_WORKERS`: 工作进程数（默认：4）
- `API_DEBUG`: 调试模式（默认：true）

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
4. 部署说明：
   - 确保 Milvus 服务已正确配置并可访问
   - 确保模型服务已正确部署并可访问
   - 根据实际环境修改相关配置参数

## 许可证

[添加许可证信息] 