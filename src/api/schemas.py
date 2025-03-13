from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class CreateDocumentRequest(BaseModel):
    """创建文档请求"""
    texts: List[str] = Field(..., description="文档内容列表")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="文档元数据列表")
    batch_size: Optional[int] = Field(32, description="批处理大小")

class CreateDocumentResponse(BaseModel):
    """创建文档响应"""
    message: str = Field("success", description="处理结果消息")
    data: Dict[str, Any] = Field(..., description="处理结果数据")

class SearchResult(BaseModel):
    """搜索结果"""
    content: str = Field(..., description="文档块内容")
    score: float = Field(..., description="混合检索分数")
    dense_score: Optional[float] = Field(default=None, description="稠密向量检索分数")
    sparse_score: Optional[float] = Field(default=None, description="稀疏向量检索分数")
    chunk_index: int = Field(default=0, description="文档块索引")

    class Config:
        schema_extra = {
            "example": {
                "content": "文档块内容...",
                "score": 0.85,
                "dense_score": 0.9,
                "sparse_score": 0.75,
                "chunk_index": 0
            }
        }

class RetrievalRequest(BaseModel):
    """检索请求"""
    query: str = Field(..., description="搜索查询")
    limit: int = Field(default=5, description="返回结果数量限制")
    filter_str: Optional[str] = Field(default=None, description="元数据过滤条件字符串")
    rerank: bool = Field(default=False, description="是否启用重排序")
    include_parent_doc: bool = Field(default=True, description="是否包含父文档信息")
    dense_weight: float = Field(default=0.7, description="稠密向量权重 (0-1)")
    sparse_weight: float = Field(default=0.3, description="稀疏向量权重 (0-1)")

    class Config:
        schema_extra = {
            "example": {
                "query": "搜索查询",
                "limit": 5,
                "filter_str": "source:doc AND created_at:>=2024-01-01",
                "rerank": False,
                "include_parent_doc": True,
                "dense_weight": 0.7,
                "sparse_weight": 0.3
            }
        }

class RetrievalResponse(BaseModel):
    """检索响应"""
    results: List[SearchResult] = Field(..., description="搜索结果列表")

class ErrorResponse(BaseModel):
    """错误响应"""
    error_code: int = Field(..., description="错误代码")
    error_msg: str = Field(..., description="错误信息") 