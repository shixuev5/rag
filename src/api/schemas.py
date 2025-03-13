from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class CreateDocumentRequest(BaseModel):
    """创建文档请求"""
    texts: List[str] = Field(..., description="文档内容列表")
    metadata: Optional[List[Dict]] = Field(None, description="文档元数据列表")
    batch_size: Optional[int] = Field(32, description="批处理大小")

class CreateDocumentResponse(BaseModel):
    """创建文档响应"""
    message: str = Field("success", description="处理结果消息")
    data: Dict = Field(..., description="处理结果数据")

class SearchResult(BaseModel):
    """搜索结果"""
    content: str = Field(..., description="文档内容")
    source: str = Field(..., description="文档来源")
    file_name: str = Field(..., description="文件名")
    file_type: str = Field(..., description="文件类型")
    title: Optional[str] = Field(None, description="文档标题")
    created_at: datetime = Field(..., description="创建时间")
    modified_at: datetime = Field(..., description="修改时间")
    score: float = Field(..., description="相关度分数")
    original_score: Optional[float] = Field(None, description="原始分数")

class RetrievalRequest(BaseModel):
    """检索请求"""
    query: str = Field(..., description="检索查询")
    limit: int = Field(5, description="返回结果数量")
    filter_str: Optional[str] = Field(None, description="过滤条件字符串")
    use_hybrid: bool = Field(True, description="是否使用混合检索")
    vector_weight: float = Field(0.7, description="向量检索权重")
    rerank: bool = Field(True, description="是否使用重排序")

class RetrievalResponse(BaseModel):
    """检索响应"""
    results: List[SearchResult] = Field(..., description="检索结果列表")

class ErrorResponse(BaseModel):
    """错误响应"""
    error_code: int = Field(..., description="错误代码")
    error_msg: str = Field(..., description="错误信息") 