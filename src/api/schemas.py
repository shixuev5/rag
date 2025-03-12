from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class VectorizeRequest(BaseModel):
    """文档向量化请求"""
    texts: List[str] = Field(..., description="待向量化的文本列表")
    batch_size: int = Field(32, description="批处理大小")

class VectorizeResponse(BaseModel):
    """文档向量化响应"""
    vectors: List[List[float]] = Field(..., description="向量化结果")
    dimensions: int = Field(..., description="向量维度")

class RetrievalSetting(BaseModel):
    """检索设置"""
    top_k: int = Field(5, description="返回结果数量")
    score_threshold: float = Field(0.5, description="相关度分数阈值(0~1)")

class RetrievalRequest(BaseModel):
    """检索请求"""
    knowledge_id: str = Field(..., description="知识库ID")
    query: str = Field(..., description="查询文本")
    retrieval_setting: RetrievalSetting = Field(..., description="检索设置")

class RecordMetadata(BaseModel):
    """文档元数据"""
    source: str = Field(..., description="文档来源")
    created_at: float = Field(..., description="创建时间")
    modified_at: float = Field(..., description="修改时间")
    extra: Dict = Field(default_factory=dict, description="额外元数据")

class RetrievalRecord(BaseModel):
    """检索结果记录"""
    content: str = Field(..., description="文档内容")
    score: float = Field(..., description="相关度分数")
    title: str = Field(..., description="文档标题")
    metadata: Optional[RecordMetadata] = Field(None, description="元数据")

class RetrievalResponse(BaseModel):
    """检索响应"""
    records: List[RetrievalRecord] = Field(..., description="检索结果列表")

class ErrorResponse(BaseModel):
    """错误响应"""
    error_code: int = Field(..., description="错误代码")
    error_msg: str = Field(..., description="错误信息") 