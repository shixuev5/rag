from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging
from datetime import datetime
from langchain.schema import Document

from .schemas import *
from core.vector_store import MilvusVectorStore
from core.ranker import Reranker
from utils.filters import parse_metadata_filters
from config.settings import (
    ERROR_CODES
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
vector_store = MilvusVectorStore()
reranker = Reranker()

@app.post("/documents", response_model=CreateDocumentResponse)
async def create_documents(request: CreateDocumentRequest):
    """添加文档接口"""
    try:
        logger.info(f"Processing document creation request with {len(request.texts)} texts")
        
        # 创建文档对象列表
        documents = []
        for i, text in enumerate(request.texts):
            metadata = request.metadata[i] if request.metadata and i < len(request.metadata) else {}
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        # 添加文档到向量存储
        vector_store.add_documents(documents)
        
        response = CreateDocumentResponse(
            message="success",
            data={"document_count": len(documents)}
        )
        
        logger.info(f"Documents creation completed with {len(documents)} documents")
        return response
        
    except Exception as e:
        logger.error(f"Documents creation failed: {str(e)}")
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
        logger.info(f"Processing retrieval request: {request.query}")
        
        # 解析过滤条件
        filters = parse_metadata_filters(request.filter_str) if request.filter_str else None
        
        # 执行初始检索
        initial_results = vector_store.search(
            query=request.query,
            limit=request.limit * 2 if request.rerank else request.limit,
            metadata_filters=filters,
            use_hybrid=request.use_hybrid,
            vector_weight=request.vector_weight
        )
        
        # 执行重排序（如果启用）
        results = reranker.rerank(request.query, initial_results, request.limit) if request.rerank else initial_results[:request.limit]
        
        response = RetrievalResponse(
            results=[
                SearchResult(
                    content=hit["content"],
                    source=hit["source"],
                    file_name=hit["file_name"],
                    file_type=hit["file_type"],
                    title=hit["title"],
                    created_at=hit["created_at"],
                    modified_at=hit["modified_at"],
                    score=hit["score"],
                    original_score=hit.get("original_score")
                )
                for hit in results
            ]
        )
        
        logger.info(f"Retrieval completed with {len(results)} results")
        return response
        
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