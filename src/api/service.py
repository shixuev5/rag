from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
from langchain.schema import Document

from .schemas import *
from core.vector_store import MilvusVectorStore
from core.ranker import Reranker
from core.retriever import Retriever
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
retriever = Retriever(vector_store, reranker)

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
    """文档检索接口
    
    Args:
        request: 检索请求，包含以下参数：
            - query: 搜索查询
            - limit: 返回结果数量限制
            - filter_str: 元数据过滤条件字符串
            - rerank: 是否启用重排序
            - include_parent_doc: 是否包含父文档信息
            - dense_weight: 稠密向量权重 (0-1)
            - sparse_weight: 稀疏向量权重 (0-1)
    """
    try:
        # 使用 Retriever 执行检索
        results = retriever.retrieve(
            query=request.query,
            limit=request.limit,
            filter_str=request.filter_str,
            rerank=request.rerank,
            include_parent_doc=request.include_parent_doc,
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight
        )
        
        # 转换结果为响应格式
        response_results = []
        for result in results:
            chunk = result["chunk"]
            
            response_results.append(
                SearchResult(
                    content=chunk["content"],
                    score=chunk["score"],
                    dense_score=chunk.get("dense_score"),
                    sparse_score=chunk.get("sparse_score"),
                    chunk_index=chunk.get("chunk_index", 0)
                )
            )
        
        response = RetrievalResponse(
            results=response_results
        )
        
        logger.info(f"Retrieval completed with {len(response_results)} results")
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