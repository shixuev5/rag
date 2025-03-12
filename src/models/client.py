import requests
from typing import List, Dict, Any
from src.config.settings import MODEL_SERVICE

class ModelClient:
    """模型服务客户端"""
    
    def __init__(self):
        self.base_url = f"http://{MODEL_SERVICE['HOST']}:{MODEL_SERVICE['PORT']}"
        self.timeout = MODEL_SERVICE['TIMEOUT']
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本向量表示
        
        Args:
            texts: 待向量化的文本列表
            
        Returns:
            向量列表
        """
        url = f"{self.base_url}{MODEL_SERVICE['EMBEDDING_PATH']}"
        
        try:
            response = requests.post(
                url,
                json={"texts": texts},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            raise Exception(f"Embedding request failed: {str(e)}")
    
    def rerank(self, query: str, documents: List[str], top_n: int) -> List[Dict[str, Any]]:
        """对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_n: 返回结果数量
            
        Returns:
            重排序结果列表，每个结果包含文档内容、索引和相关度分数
        """
        url = f"{self.base_url}{MODEL_SERVICE['RERANK_PATH']}"
        
        try:
            response = requests.post(
                url,
                json={
                    "model": RERANK_MODEL_NAME,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["results"]
        except Exception as e:
            raise Exception(f"Rerank request failed: {str(e)}") 