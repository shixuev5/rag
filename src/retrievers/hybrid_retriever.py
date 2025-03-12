from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from rank_bm25 import BM25Okapi
from langchain.schema import Document

from src.retrievers.base import BaseRetriever
from src.models.client import ModelClient
from src.config.settings import (
    MILVUS_HOST,
    MILVUS_PORT,
    COLLECTION_NAME,
    VECTOR_DIM,
    DEFAULT_VECTOR_WEIGHT
)

class HybridRetriever(BaseRetriever):
    def __init__(self):
        self.model_client = ModelClient()
        self._connect_milvus()
        self._init_collection()
        self.bm25 = None
        self.documents = []

    def _connect_milvus(self):
        """连接到Milvus服务器"""
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )

    def _init_collection(self):
        """初始化Milvus集合"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="created_at", dtype=DataType.DOUBLE),
            FieldSchema(name="modified_at", dtype=DataType.DOUBLE)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Document collection for hybrid search"
        )

        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)

        self.collection = Collection(name=COLLECTION_NAME, schema=schema)
        
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.collection.create_index(field_name="vector", index_params=index_params)

    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到检索器"""
        if not documents:
            return

        # 保存文档用于BM25检索
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.bm25 = BM25Okapi([text.split() for text in texts])

        # 获取文档向量
        vectors = self.model_client.get_embeddings(texts)

        # 准备数据
        sources = [doc.metadata.get("source", "") for doc in documents]
        file_names = [doc.metadata.get("file_name", "") for doc in documents]
        file_types = [doc.metadata.get("file_type", "") for doc in documents]
        titles = [doc.metadata.get("title", "") for doc in documents]
        created_ats = [doc.metadata.get("created_at", 0.0) for doc in documents]
        modified_ats = [doc.metadata.get("modified_at", 0.0) for doc in documents]

        entities = [
            texts,
            vectors,
            sources,
            file_names,
            file_types,
            titles,
            created_ats,
            modified_ats
        ]

        self.collection.insert(entities)
        self.collection.flush()

    def search(self,
              query: str,
              limit: int,
              metadata_filters: Optional[Dict[str, Any]] = None,
              use_hybrid: bool = True,
              vector_weight: float = DEFAULT_VECTOR_WEIGHT) -> List[Dict]:
        """混合搜索文档"""
        query_vector = self.model_client.get_embeddings([query])[0]
        self.collection.load()
        
        filter_expr = self._build_filter_expr(metadata_filters) if metadata_filters else None
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        vector_results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=limit if not use_hybrid else limit * 2,
            expr=filter_expr,
            output_fields=["content", "source", "file_name", "file_type", "title", 
                          "created_at", "modified_at"]
        )

        if not use_hybrid or not self.bm25:
            return self._format_results(vector_results[0])

        # BM25检索
        bm25_scores = self.bm25.get_scores(query.split())
        
        # 合并结果
        hybrid_results = []
        for hit in vector_results[0]:
            idx = self._find_document_index(hit.entity.get("content"))
            if idx != -1:
                vector_score = 1 - hit.score
                bm25_score = bm25_scores[idx]
                normalized_bm25_score = bm25_score / max(bm25_scores) if max(bm25_scores) > 0 else 0
                hybrid_score = vector_weight * vector_score + (1 - vector_weight) * normalized_bm25_score
                hybrid_results.append((hit, hybrid_score))

        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        return self._format_results([hit for hit, _ in hybrid_results[:limit]])

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