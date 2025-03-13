from typing import List, Dict
from models.client import ModelClient
from config.settings import FINAL_RETRIEVAL_SIZE

class Reranker:
    """重排序器"""
    
    def __init__(self):
        self.model_client = ModelClient()
        
    def rerank(self, query: str, results: List[Dict], limit: int = FINAL_RETRIEVAL_SIZE) -> List[Dict]:
        """对检索结果进行重排序"""
        if not results:
            return []
            
        # 准备重排序输入
        documents = [hit["content"] for hit in results]
        
        # 调用重排序服务
        rerank_results = self.model_client.rerank(query, documents, limit)
        
        # 更新结果分数
        for rerank_result, result in zip(rerank_results, results):
            result["original_score"] = result["score"]
            result["score"] = rerank_result["relevance_score"]
            
        # 按新分数排序并返回指定数量的结果
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit] 