from typing import List, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod
import markdown
from bs4 import BeautifulSoup
from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

class BaseDocumentProcessor(ABC):
    """文档处理抽象基类"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def process_file(self, file_path: Path) -> List[Document]:
        """处理单个文件"""
        pass

    def process_directory(self, directory: Path) -> List[Document]:
        """处理目录中的文件"""
        all_docs = []
        for file_path in self._get_files(directory):
            docs = self.process_file(file_path)
            all_docs.extend(docs)
        return all_docs

    @abstractmethod
    def _get_files(self, directory: Path) -> List[Path]:
        """获取目录中的目标文件"""
        pass

    def _add_metadata(self, doc: Document, file_path: Path, additional_metadata: Dict[str, Any] = None) -> Document:
        """添加元数据"""
        metadata = {
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": file_path.suffix.lower()[1:],
            "created_at": file_path.stat().st_ctime,
            "modified_at": file_path.stat().st_mtime,
        }
        if additional_metadata:
            metadata.update(additional_metadata)
        doc.metadata.update(metadata)
        return doc

class MarkdownProcessor(BaseDocumentProcessor):
    """Markdown文档处理器"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        super().__init__(chunk_size, chunk_overlap)
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_file(self, file_path: Path) -> List[Document]:
        """处理Markdown文件并返回文档片段"""
        # 读取Markdown文件
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # 将Markdown转换为HTML以便提取结构化内容
        html_content = markdown.markdown(md_content, extensions=['meta'])
        soup = BeautifulSoup(html_content, 'html.parser')

        # 提取元数据（如果有的话）
        metadata = {}
        headers = soup.find_all(['h1', 'h2', 'h3'])
        if headers:
            metadata['title'] = headers[0].get_text()
            metadata['headers'] = [h.get_text() for h in headers]

        # 提取纯文本
        text = soup.get_text()

        # 分割文本
        docs = self.text_splitter.create_documents([text])
        
        # 添加元数据
        for doc in docs:
            self._add_metadata(doc, file_path, metadata)

        return docs

    def _get_files(self, directory: Path) -> List[Path]:
        """获取目录中的Markdown文件"""
        return list(directory.glob("**/*.md")) 