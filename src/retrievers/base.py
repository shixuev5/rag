from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseRetriever(ABC):
    """检索器基类"""
    
    @abstractmethod
    def add_documents(self, documents: List[Any]) -> None:
        """添加文档到检索器"""
        pass
    
    @abstractmethod
    def search(self, 
              query: str,
              limit: int,
              metadata_filters: Optional[Dict[str, Any]] = None,
              **kwargs) -> List[Dict]:
        """搜索文档"""
        pass 