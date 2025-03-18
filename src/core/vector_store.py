from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document
from models.client import ModelClient
from utils.text_preprocessor import TextPreprocessor
from config.settings import (
    MILVUS_HOST,
    MILVUS_PORT,
    VECTOR_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
import uuid
import logging
from typing import Optional, List, Dict, Any
from pymilvus.exceptions import MilvusException

# 集合名称常量
DOCUMENTS_COLLECTION_NAME = "documents"
CHUNKS_COLLECTION_NAME = "document_chunks"

logger = logging.getLogger(__name__)

class VectorStore(ABC):
    """向量存储基类"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量存储"""
        pass
    
    @abstractmethod
    def search_chunks(self,
              query: str,
              limit: int = 5,
              metadata_filters: Optional[Dict[str, Any]] = None,
              include_parent_doc: bool = True,
              **kwargs) -> List[Dict]:
        """搜索文档块
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            metadata_filters: 元数据过滤条件
            include_parent_doc: 是否包含父文档信息
            **kwargs: 其他参数
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        """删除文档"""
        pass

    @abstractmethod
    def update_document(self, document_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """更新文档"""
        pass

class MilvusVectorStore(VectorStore):
    def __init__(self):
        self.model_client = ModelClient()
        self.text_preprocessor = TextPreprocessor()
        self._connect_milvus()
        self._init_collections()

    def __del__(self):
        """析构函数，确保关闭连接"""
        try:
            connections.disconnect("default")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")

    def _connect_milvus(self):
        """连接到Milvus服务器"""
        try:
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT
            )
        except MilvusException as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _init_collections(self):
        """初始化文档和文档块集合"""
        try:
            # 文档集合字段
            document_fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
                FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunks", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=1024, max_length=64),  # 添加 max_length 参数
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="created_at", dtype=DataType.DOUBLE),
                FieldSchema(name="modified_at", dtype=DataType.DOUBLE),
            ]

            # 文档块集合字段
            chunk_fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
                FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),
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
                # 为新创建的文档集合创建向量索引
                doc_dense_index_params = {
                    "metric_type": "IP",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                doc_sparse_index_params = {
                    "metric_type": "IP",
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "params": {}
                }
                self.documents_collection.create_index(field_name="dense", index_params=doc_dense_index_params)
                self.documents_collection.create_index(field_name="sparse", index_params=doc_sparse_index_params)

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
                    "metric_type": "IP",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                self.chunks_collection.create_index(field_name="dense", index_params=index_params)
                
                # 为稀疏向量字段创建索引
                sparse_index_params = {
                    "metric_type": "IP",
                    "index_type": "SPARSE_INVERTED_INDEX",  # 使用稀疏向量专用的索引类型
                    "params": {}
                }
                self.chunks_collection.create_index(field_name="sparse", index_params=sparse_index_params)

            # 加载集合
            self.documents_collection.load()
            self.chunks_collection.load()
        except MilvusException as e:
            logger.error(f"Failed to initialize collections: {e}")
            raise

    def _convert_to_sparse_matrix_format(self, sparse_vector):
        """将稀疏向量转换为Milvus支持的格式
        
        Milvus期望的稀疏向量格式为：
        {
            "indices": [0, 2, 4],  # 非零元素的索引
            "values": [0.1, -0.2, 0.4]  # 对应的非零值
        }
        """
        if isinstance(sparse_vector, dict) and "indices" in sparse_vector and "values" in sparse_vector:
            return sparse_vector
            
        # 如果是普通列表，转换为稀疏格式
        indices = []
        values = []
        for idx, value in enumerate(sparse_vector):
            if value != 0:
                indices.append(idx)
                values.append(float(value))
                
        return {
            "indices": indices,
            "values": values
        }

    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量存储"""
        if not documents:
            return

        try:
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
                chunk_denses = []
                chunk_sparses = []
                chunk_summaries = []
                chunk_indices = []
                chunk_metadata_list = []
                chunk_created_ats = []
                
                # 处理每个分块
                for i, chunk in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())
                    chunk_ids.append(chunk_id)
                    
                    # 获取分块向量
                    dense_vector = self.model_client.get_embeddings([chunk], embedding_type="dense_vecs")[0]
                    
                    # 预处理文本并获取稀疏向量
                    processed_text = self.text_preprocessor.preprocess(chunk)
                    sparse_vector = self.model_client.get_embeddings([processed_text], embedding_type="sparse_vecs")[0]
                    sparse_vector = self._convert_to_sparse_matrix_format(sparse_vector)
                    
                    # 生成主题概述
                    summary = self.model_client.summarize(chunk)
                    
                    # 添加到实体列表
                    chunk_document_ids.append(doc_id)
                    chunk_contents.append(chunk)
                    chunk_denses.append(dense_vector)
                    chunk_sparses.append(sparse_vector)
                    chunk_summaries.append(summary)
                    chunk_indices.append(i)
                    chunk_metadata_list.append(doc.metadata)
                    chunk_created_ats.append(datetime.now().timestamp())
                
                # 批量插入文档块
                self.chunks_collection.insert([
                    chunk_ids,
                    chunk_document_ids,
                    chunk_contents,
                    chunk_denses,
                    chunk_sparses,
                    chunk_summaries,
                    chunk_indices,
                    chunk_metadata_list,
                    chunk_created_ats
                ])
                
                # 合并所有 chunks 的摘要并生成文档摘要
                all_chunk_summaries = "\n\n".join([chunk_summaries[i] for i in range(len(chunks))])

                # 获取文档级别的向量
                doc_dense_vector = self.model_client.get_embeddings([all_chunk_summaries], embedding_type="dense_vecs")[0]
                doc_sparse_vector = self.model_client.get_embeddings([all_chunk_summaries], embedding_type="sparse_vecs")[0]

                doc_summary = self.model_client.summarize(all_chunk_summaries)

                # 保存原始文档
                current_time = datetime.now().timestamp()
                
                document_entity = {
                    "id": doc_id,
                    "content": doc.page_content,
                    "dense": doc_dense_vector,
                    "sparse": doc_sparse_vector,
                    "summary": doc_summary,
                    "chunks": chunk_ids,
                    "metadata": doc.metadata,
                    "created_at": current_time,
                    "modified_at": current_time,
                }
                
                self.documents_collection.insert([
                    [document_entity["id"]],
                    [document_entity["content"]],
                    [document_entity["dense"]],
                    [document_entity["sparse"]],
                    [document_entity["summary"]],
                    [document_entity["chunks"]],
                    [document_entity["metadata"]],
                    [document_entity["created_at"]],
                    [document_entity["modified_at"]],
                ])
                
            # 刷新集合
            self.documents_collection.flush()
            self.chunks_collection.flush()
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def search_chunks(self,
              query: str,
              limit: int = 5,
              metadata_filters: Optional[Dict[str, Any]] = None,
              include_parent_doc: bool = True,
              dense_weight: float = 0.7,
              sparse_weight: float = 0.3,
              **kwargs) -> List[Dict]:
        """混合搜索文档块并关联原始文档信息
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            metadata_filters: 元数据过滤条件
            include_parent_doc: 是否包含父文档信息
            dense_weight: 稠密向量权重 (0-1)
            sparse_weight: 稀疏向量权重 (0-1)
            **kwargs: 其他参数
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            # 参数验证
            if not 0 <= dense_weight <= 1 or not 0 <= sparse_weight <= 1:
                raise ValueError("Weights must be between 0 and 1")
            if abs(dense_weight + sparse_weight - 1.0) > 1e-6:
                raise ValueError("Sum of weights must be 1.0")
            
            # 获取查询向量
            query_vector = self.model_client.get_embeddings([query])[0]
            
            # 加载集合
            self.chunks_collection.load()
            if include_parent_doc:
                self.documents_collection.load()
            
            # 构建过滤表达式
            filter_expr = self._build_filter_expr(metadata_filters) if metadata_filters else None
            
            # 稠密向量搜索参数
            dense_search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
            
            # 稀疏向量搜索参数
            sparse_search_params = {
                "metric_type": "BM25",
                "params": {"nprobe": 10}
            }
            
            # 执行稠密向量搜索
            dense_results = self.chunks_collection.search(
                data=[query_vector],
                anns_field="dense",
                param=dense_search_params,
                limit=limit * 2,  # 获取更多结果用于混合排序
                expr=filter_expr,
                output_fields=["document_id", "content", "summary", "chunk_index", "metadata", "created_at"]
            )
            
            # 执行稀疏向量搜索
            sparse_results = self.chunks_collection.search(
                data=[query_vector],
                anns_field="sparse",
                param=sparse_search_params,
                limit=limit * 2,  # 获取更多结果用于混合排序
                expr=filter_expr,
                output_fields=["document_id", "content", "summary", "chunk_index", "metadata", "created_at"]
            )
            
            # 合并和排序结果
            chunk_scores = {}
            
            # 处理稠密向量结果
            for hit in dense_results[0]:
                chunk_id = hit.entity.get("document_id")
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        "hit": hit,
                        "dense_score": hit.score,
                        "sparse_score": 0.0
                    }
            
            # 处理稀疏向量结果
            for hit in sparse_results[0]:
                chunk_id = hit.entity.get("document_id")
                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id]["sparse_score"] = hit.score
                else:
                    chunk_scores[chunk_id] = {
                        "hit": hit,
                        "dense_score": 0.0,
                        "sparse_score": hit.score
                    }
            
            # 计算混合分数并排序
            mixed_results = []
            for chunk_id, scores in chunk_scores.items():
                mixed_score = (
                    dense_weight * scores["dense_score"] +
                    sparse_weight * scores["sparse_score"]
                )
                mixed_results.append((mixed_score, scores["hit"]))
            
            # 按混合分数排序并限制结果数量
            mixed_results.sort(key=lambda x: x[0], reverse=True)
            mixed_results = mixed_results[:limit]
            
            # 如果不需要父文档信息，直接返回文档块结果
            if not include_parent_doc:
                return [{
                    "chunk": {
                        "content": hit.entity.get("content"),
                        "summary": hit.entity.get("summary"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "metadata": hit.entity.get("metadata", {}),
                        "score": score,
                        "dense_score": chunk_scores[hit.entity.get("document_id")]["dense_score"],
                        "sparse_score": chunk_scores[hit.entity.get("document_id")]["sparse_score"],
                        "created_at": datetime.fromtimestamp(hit.entity.get("created_at", 0))
                    }
                } for score, hit in mixed_results]

            # 获取相关文档块的文档ID
            document_ids = list(set(hit.entity.get("document_id") for score, hit in mixed_results))
            
            # 查询原始文档信息
            documents = {}
            if document_ids:
                doc_expr = f'id in {document_ids}'
                document_results = self.documents_collection.query(
                    expr=doc_expr,
                    output_fields=["id", "content", "summary", "metadata", "created_at", "modified_at"]
                )
                documents = {doc["id"]: doc for doc in document_results}

            # 合并结果
            results = []
            for score, hit in mixed_results:
                doc_id = hit.entity.get("document_id")
                doc_info = documents.get(doc_id, {})
                
                results.append({
                    "chunk": {
                        "content": hit.entity.get("content"),
                        "summary": hit.entity.get("summary"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "metadata": hit.entity.get("metadata", {}),
                        "score": score,
                        "dense_score": chunk_scores[doc_id]["dense_score"],
                        "sparse_score": chunk_scores[doc_id]["sparse_score"],
                        "created_at": datetime.fromtimestamp(hit.entity.get("created_at", 0))
                    },
                    "document": {
                        "id": doc_id,
                        "content": doc_info.get("content", ""),
                        "summary": doc_info.get("summary", ""),
                        "metadata": doc_info.get("metadata", {}),
                        "created_at": datetime.fromtimestamp(doc_info.get("created_at", 0)),
                        "modified_at": datetime.fromtimestamp(doc_info.get("modified_at", 0))
                    }
                })
                
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    def _build_filter_expr(self, filters: Dict[str, Any]) -> str:
        """构建Milvus过滤表达式"""
        if not filters:
            return None
            
        expressions = []
        for field, value in filters.items():
            if isinstance(value, (list, tuple)):
                # 使用参数化查询防止SQL注入
                value_list = [f"'{v}'" if isinstance(v, str) else str(v) for v in value]
                expressions.append(f"{field} in [{','.join(value_list)}]")
            elif isinstance(value, dict):
                for op, val in value.items():
                    if op == "gte":
                        expressions.append(f"{field} >= {val}")
                    elif op == "lte":
                        expressions.append(f"{field} <= {val}")
                    elif op == "contains":
                        # 使用参数化查询防止SQL注入
                        expressions.append(f"{field} like '%{val.replace('%', '%%')}%'")
            else:
                # 使用参数化查询防止SQL注入
                if isinstance(value, str):
                    expressions.append(f"{field} == '{value.replace(chr(39), chr(39)+chr(39))}'")
                else:
                    expressions.append(f"{field} == {value}")
        return " and ".join(expressions)

    def delete_documents(self, document_ids: List[str]) -> None:
        """删除文档及其关联的所有文档块"""
        try:
            # 删除文档块
            chunks_expr = f"document_id in {document_ids}"
            self.chunks_collection.delete(chunks_expr)
            
            # 删除原始文档
            docs_expr = f"id in {document_ids}"
            self.documents_collection.delete(docs_expr)
            
            # 刷新集合
            self.chunks_collection.flush()
            self.documents_collection.flush()
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def update_document(self, document_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """更新文档及其关联的文档块"""
        try:
            # 检查文档是否存在
            doc_results = self.documents_collection.query(
                expr=f"id == '{document_id}'",
                output_fields=["id", "chunks"]
            )
            if not doc_results:
                raise ValueError(f"Document with id {document_id} not found")

            # 获取旧的文档块ID列表
            old_chunk_ids = doc_results[0].get("chunks", [])

            # 初始化文档处理器
            splitter = MarkdownTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )

            # 分割新文档
            chunks = splitter.split_text(content)
            
            # 处理并保存新的文档块
            chunk_ids = []
            chunk_document_ids = []
            chunk_contents = []
            chunk_denses = []
            chunk_sparses = []
            chunk_summaries = []
            chunk_indices = []
            chunk_metadata_list = []
            chunk_created_ats = []
            
            # 处理每个分块
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                
                # 获取分块向量
                dense_vector = self.model_client.get_embeddings([chunk], embedding_type="dense_vecs")[0]
                
                # 预处理文本并获取稀疏向量
                processed_text = self.text_preprocessor.preprocess(chunk)
                sparse_vector = self.model_client.get_embeddings([processed_text], embedding_type="sparse_vecs")[0]
                sparse_vector = self._convert_to_sparse_matrix_format(sparse_vector)
                
                # 生成主题概述
                summary = self.model_client.summarize(chunk)
                
                # 添加到实体列表
                chunk_document_ids.append(document_id)
                chunk_contents.append(chunk)
                chunk_denses.append(dense_vector)
                chunk_sparses.append(sparse_vector)
                chunk_summaries.append(summary)
                chunk_indices.append(i)
                chunk_metadata_list.append(metadata or {})
                chunk_created_ats.append(datetime.now().timestamp())

            # 删除旧的文档块
            if old_chunk_ids:
                chunks_expr = f"id in {old_chunk_ids}"
                self.chunks_collection.delete(chunks_expr)
                self.chunks_collection.flush()

            # 插入新的文档块
            self.chunks_collection.insert([
                chunk_ids,
                chunk_document_ids,
                chunk_contents,
                chunk_denses,
                chunk_sparses,
                chunk_summaries,
                chunk_indices,
                chunk_metadata_list,
                chunk_created_ats
            ])
            
            # 合并所有 chunks 的摘要并生成文档摘要
            all_chunk_summaries = "\n\n".join(chunk_summaries)

            # 获取文档级别的向量
            doc_dense_vector = self.model_client.get_embeddings([all_chunk_summaries], embedding_type="dense_vecs")[0]
            doc_sparse_vector = self.model_client.get_embeddings([all_chunk_summaries], embedding_type="sparse_vecs")[0]

            doc_summary = self.model_client.summarize(all_chunk_summaries)

            # 更新文档内容
            current_time = datetime.now().timestamp()
            
            update_data = {
                "content": content,
                "dense": doc_dense_vector,
                "sparse": doc_sparse_vector,
                "summary": doc_summary,
                "chunks": chunk_ids,
                "modified_at": current_time
            }
            if metadata is not None:
                update_data["metadata"] = metadata

            self.documents_collection.update(
                expr=f"id == '{document_id}'",
                data=update_data
            )

            # 刷新集合
            self.chunks_collection.flush()
            self.documents_collection.flush()
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise 