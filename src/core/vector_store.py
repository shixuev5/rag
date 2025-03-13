from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from rank_bm25 import BM25Okapi
from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document
from models.client import ModelClient
from config.settings import (
    MILVUS_HOST,
    MILVUS_PORT,
    VECTOR_DIM,
    DEFAULT_VECTOR_WEIGHT,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
import uuid

# 集合名称常量
DOCUMENTS_COLLECTION_NAME = "documents"
CHUNKS_COLLECTION_NAME = "document_chunks"

class VectorStore(ABC):
    """向量存储基类"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量存储"""
        pass
    
    @abstractmethod
    def search(self,
              query: str,
              limit: int,
              metadata_filters: Optional[Dict[str, Any]] = None,
              **kwargs) -> List[Dict]:
        """搜索文档"""
        pass
    
    @abstractmethod
    def delete(self, document_ids: List[str]) -> None:
        """删除文档"""
        pass 

class MilvusVectorStore(VectorStore):
    def __init__(self):
        self.model_client = ModelClient()
        self._connect_milvus()
        self._init_collections()

    def _connect_milvus(self):
        """连接到Milvus服务器"""
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )

    def _init_collections(self):
        """初始化文档和文档块集合"""
        # 文档集合字段
        document_fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunks", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=1024),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.DOUBLE),
            FieldSchema(name="modified_at", dtype=DataType.DOUBLE)
        ]

        # 文档块集合字段
        chunk_fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.DOUBLE)
        ]

        # 创建文档集合
        document_schema = CollectionSchema(
            fields=document_fields,
            description="Original documents collection"
        )

        # 创建文档块集合
        chunk_schema = CollectionSchema(
            fields=chunk_fields,
            description="Document chunks collection"
        )

        # 获取或创建集合
        if utility.has_collection(DOCUMENTS_COLLECTION_NAME):
            self.documents_collection = Collection(DOCUMENTS_COLLECTION_NAME)
        else:
            self.documents_collection = Collection(
                name=DOCUMENTS_COLLECTION_NAME,
                schema=document_schema,
                using='default'
            )

        if utility.has_collection(CHUNKS_COLLECTION_NAME):
            self.chunks_collection = Collection(CHUNKS_COLLECTION_NAME)
        else:
            self.chunks_collection = Collection(
                name=CHUNKS_COLLECTION_NAME,
                schema=chunk_schema,
                using='default'
            )
            # 为新创建的文档块集合创建向量索引
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.chunks_collection.create_index(field_name="vector", index_params=index_params)

        # 加载集合
        self.documents_collection.load()
        self.chunks_collection.load()

    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量存储"""
        if not documents:
            return

        # 初始化文档处理器
        splitter = MarkdownTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # 处理每个文档
        for doc in documents:
            # 生成文档唯一ID
            doc_id = str(uuid.uuid4())
            
            # 分割文档
            chunks = splitter.split_text(doc.page_content)
            
            # 处理并保存文档块
            chunk_ids = []
            chunk_document_ids = []
            chunk_contents = []
            chunk_vectors = []
            chunk_keywords_list = []
            chunk_summaries = []
            chunk_indices = []
            chunk_metadata_list = []
            chunk_created_ats = []
            
            # 处理每个分块
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                
                # 获取分块向量
                chunk_vector = self.model_client.get_embeddings([chunk])[0]
                
                # 提取关键词 (使用 BM25)
                tokenized_text = chunk.split()
                bm25 = BM25Okapi([tokenized_text])
                keywords = [word for word, _ in sorted(
                    [(word, score) for word, score in zip(set(tokenized_text), bm25.get_scores(tokenized_text))],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]]  # 取前10个关键词
                
                # 生成主题概述
                summary = self.model_client.summarize(chunk)
                
                # 添加到实体列表
                chunk_document_ids.append(doc_id)
                chunk_contents.append(chunk)
                chunk_vectors.append(chunk_vector)
                chunk_keywords_list.append(",".join(keywords))
                chunk_summaries.append(summary)
                chunk_indices.append(i)
                chunk_metadata_list.append(doc.metadata)
                chunk_created_ats.append(datetime.now().timestamp())
            
            # 批量插入文档块
            self.chunks_collection.insert([
                chunk_ids,
                chunk_document_ids,
                chunk_contents,
                chunk_vectors,
                chunk_keywords_list,
                chunk_summaries,
                chunk_indices,
                chunk_metadata_list,
                chunk_created_ats
            ])
            
            # 保存原始文档
            current_time = datetime.now().timestamp()
            document_entity = {
                "id": doc_id,
                "content": doc.page_content,
                "chunks": chunk_ids,
                "metadata": doc.metadata,
                "created_at": current_time,
                "modified_at": current_time
            }
            
            self.documents_collection.insert([
                [document_entity["id"]],
                [document_entity["content"]],
                [document_entity["chunks"]],
                [document_entity["metadata"]],
                [document_entity["created_at"]],
                [document_entity["modified_at"]]
            ])
            
        # 刷新集合
        self.documents_collection.flush()
        self.chunks_collection.flush()

    def search(self,
              query: str,
              limit: int = 5,
              metadata_filters: Optional[Dict[str, Any]] = None,
              use_hybrid: bool = True,
              vector_weight: float = DEFAULT_VECTOR_WEIGHT) -> List[Dict]:
        """混合搜索文档块并关联原始文档信息"""
        # 获取查询向量
        query_vector = self.model_client.get_embeddings([query])[0]
        
        # 加载集合
        self.chunks_collection.load()
        self.documents_collection.load()
        
        # 构建过滤表达式
        filter_expr = self._build_filter_expr(metadata_filters) if metadata_filters else None
        
        # 搜索文档块
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        chunk_results = self.chunks_collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=limit,
            expr=filter_expr,
            output_fields=["document_id", "content", "keywords", "summary", "chunk_index", "metadata", "created_at"]
        )

        # 获取相关文档块的文档ID
        document_ids = list(set(hit.entity.get("document_id") for hit in chunk_results[0]))
        
        # 查询原始文档信息
        documents = {}
        if document_ids:
            doc_expr = f'id in {document_ids}'
            document_results = self.documents_collection.query(
                expr=doc_expr,
                output_fields=["id", "content", "metadata", "created_at", "modified_at"]
            )
            documents = {doc["id"]: doc for doc in document_results}

        # 合并结果
        results = []
        for hit in chunk_results[0]:
            doc_id = hit.entity.get("document_id")
            doc_info = documents.get(doc_id, {})
            
            results.append({
                "chunk": {
                    "content": hit.entity.get("content"),
                    "keywords": hit.entity.get("keywords", "").split(","),
                    "summary": hit.entity.get("summary"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "metadata": hit.entity.get("metadata", {}),
                    "score": hit.score,
                    "created_at": datetime.fromtimestamp(hit.entity.get("created_at", 0))
                },
                "document": {
                    "id": doc_id,
                    "content": doc_info.get("content", ""),
                    "metadata": doc_info.get("metadata", {}),
                    "created_at": datetime.fromtimestamp(doc_info.get("created_at", 0)),
                    "modified_at": datetime.fromtimestamp(doc_info.get("modified_at", 0))
                }
            })
            
        return results

    def _build_filter_expr(self, filters: Dict[str, Any]) -> str:
        """构建Milvus过滤表达式"""
        if not filters:
            return None
            
        expressions = []
        for field, value in filters.items():
            if isinstance(value, (list, tuple)):
                expressions.append(f"{field} in {value}")
            elif isinstance(value, dict):
                for op, val in value.items():
                    if op == "gte":
                        expressions.append(f"{field} >= {val}")
                    elif op == "lte":
                        expressions.append(f"{field} <= {val}")
                    elif op == "contains":
                        expressions.append(f"{field} like '%{val}%'")
            else:
                expressions.append(f"{field} == '{value}'")
        return " and ".join(expressions)

    def _find_document_index(self, content: str) -> int:
        """查找文档在原始文档列表中的索引"""
        for i, doc in enumerate(self.documents):
            if doc.page_content == content:
                return i
        return -1

    def _format_results(self, hits) -> List[Dict]:
        """格式化搜索结果"""
        formatted_hits = []
        for hit in hits:
            formatted_hits.append({
                "content": hit.entity.get("content"),
                "source": hit.entity.get("source"),
                "file_name": hit.entity.get("file_name"),
                "file_type": hit.entity.get("file_type"),
                "title": hit.entity.get("title"),
                "created_at": datetime.fromtimestamp(hit.entity.get("created_at", 0)),
                "modified_at": datetime.fromtimestamp(hit.entity.get("modified_at", 0)),
                "score": hit.score
            })
        return formatted_hits

    def delete(self, document_ids: List[str]) -> None:
        """删除文档及其关联的所有文档块"""
        # 删除文档块
        chunks_expr = f"document_id in {document_ids}"
        self.chunks_collection.delete(chunks_expr)
        
        # 删除原始文档
        docs_expr = f"id in {document_ids}"
        self.documents_collection.delete(docs_expr)
        
        # 刷新集合
        self.chunks_collection.flush()
        self.documents_collection.flush() 