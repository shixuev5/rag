from typing import List, Dict
from src.models.client import ModelClient
from src.config.settings import FINAL_RETRIEVAL_SIZE

class Reranker:
    def __init__(self):
        self.model_client = ModelClient()

    def rerank(self, query: str, results: List[Dict], top_k: int = FINAL_RETRIEVAL_SIZE) -> List[Dict]:
        """对检索结果进行重排序
        
        Args:
            query: 查询文本
            results: 初始检索结果
            top_k: 返回结果数量
        
        Returns:
            重排序后的结果列表
        """
        if not results:
            return []

        # 准备重排序输入
        documents = [hit["content"] for hit in results]
        
        # 调用重排序服务
        rerank_results = self.model_client.rerank(query, documents, top_k)
        
        # 更新结果分数
        for rerank_result, result in zip(rerank_results, results):
            result["rerank_score"] = rerank_result["relevance_score"]
            result["original_score"] = result["score"]
            result["score"] = rerank_result["relevance_score"]
        
        # 按新分数排序并返回top_k个结果
        reranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_results[:top_k] 