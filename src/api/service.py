from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging
from datetime import datetime

from .schemas import *
from src.retrievers.hybrid_retriever import HybridRetriever
from src.config.settings import (
    ERROR_CODES,
    VECTORIZE_BATCH_SIZE
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="RAG Service API",
    description="文档向量化和检索服务API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
retriever = HybridRetriever()

@app.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_texts(request: VectorizeRequest):
    """文本向量化接口"""
    try:
        logger.info(f"Processing vectorization request with {len(request.texts)} texts")
        
        # 使用请求中的batch_size或默认值
        batch_size = request.batch_size or VECTORIZE_BATCH_SIZE
        
        # 批量处理文本向量化
        vectors = []
        for i in range(0, len(request.texts), batch_size):
            batch = request.texts[i:i + batch_size]
            batch_vectors = retriever.model_client.get_embeddings(batch)
            vectors.extend(batch_vectors)
        
        response = VectorizeResponse(
            vectors=vectors,
            dimensions=len(vectors[0]) if vectors else 0
        )
        
        logger.info(f"Vectorization completed with {len(vectors)} vectors")
        return response
        
    except Exception as e:
        logger.error(f"Vectorization failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code=3001,
                error_msg=f"{ERROR_CODES[3001]}: {str(e)}"
            ).dict()
        )

@app.post("/retrieval", response_model=RetrievalResponse)
async def retrieve_documents(request: RetrievalRequest):
    """文档检索接口"""
    try:
        logger.info(f"Processing retrieval request for knowledge_id: {request.knowledge_id}")
        
        # 验证知识库是否存在
        if not retriever.knowledge_exists(request.knowledge_id):
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error_code=2001,
                    error_msg=ERROR_CODES[2001]
                ).dict()
            )
        
        # 执行检索
        results = retriever.search(
            query=request.query,
            limit=request.retrieval_setting.top_k,
            metadata_filters={"knowledge_id": request.knowledge_id},
            score_threshold=request.retrieval_setting.score_threshold
        )
        
        # 转换为API响应格式
        records = []
        for result in results:
            metadata = RecordMetadata(
                source=result["source"],
                created_at=result["created_at"].timestamp(),
                modified_at=result["modified_at"].timestamp(),
                extra={
                    "file_type": result["file_type"],
                    "file_name": result["file_name"]
                }
            )
            
            record = RetrievalRecord(
                content=result["content"],
                score=result["score"],
                title=result.get("title", ""),
                metadata=metadata
            )
            records.append(record)
        
        response = RetrievalResponse(records=records)
        logger.info(f"Retrieval completed with {len(records)} results")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code=3002,
                error_msg=f"{ERROR_CODES[3002]}: {str(e)}"
            ).dict()
        )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    } 