import requests
from typing import List, Dict, Any
from config.settings import MODEL_SERVICE

class ModelClient:
    """模型服务客户端"""
    
    def __init__(self):
        self.base_url = f"https://{MODEL_SERVICE['HOST']}:{MODEL_SERVICE['PORT']}"
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

    def summarize(self, text: str) -> str:
        """生成文本摘要
        
        Args:
            text: 待总结的文本
            
        Returns:
            文本摘要，长度不超过 50 tokens
        """
        url = f"{self.base_url}{MODEL_SERVICE['CHAT_PATH']}"
        
        # 构建提示词
        system_prompt = "你是一个专业的文本摘要助手。你的任务是生成一个极其简洁的摘要，要求：\n1. 摘要必须控制在50个token以内\n2. 保留原文最核心的观点\n3. 使用简洁的语言\n4. 确保语义完整"
        
        user_prompt = f"请为以下文本生成一个不超过50个token的极简摘要。注意：必须确保摘要简短精炼。\n\n{text}"
        
        try:
            response = requests.post(
                url,
                json={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 50
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise Exception(f"Summarize request failed: {str(e)}") 