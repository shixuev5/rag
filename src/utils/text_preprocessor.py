import re
from typing import Set
import jieba
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """文本预处理工具类"""
    
    def __init__(self):
        """初始化预处理器，加载停用词"""
        # 加载英文停用词
        self.en_stopwords = self._load_english_stopwords()
        
        # 加载中文停用词
        self.zh_stopwords = self._load_chinese_stopwords()
        
    def _load_english_stopwords(self) -> Set[str]:
        """从文件加载英文停用词
        
        Returns:
            停用词集合
        """
        stopwords_set = set()
        
        # 获取停用词文件路径
        current_dir = Path(__file__).parent
        stopwords_file = current_dir / "stopwords" / "english_stopwords.txt"
        
        try:
            if stopwords_file.exists():
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    # 读取每行并去除空白字符
                    words = {line.strip() for line in f if line.strip()}
                    stopwords_set.update(words)
                    logger.info(f"从 {stopwords_file} 加载了 {len(words)} 个英文停用词")
            else:
                logger.warning(f"英文停用词文件 {stopwords_file} 不存在")
        except Exception as e:
            logger.error(f"加载英文停用词文件时出错: {e}")
            
        return stopwords_set
        
    def _load_chinese_stopwords(self) -> Set[str]:
        """从文件加载中文停用词
        
        Returns:
            停用词集合
        """
        stopwords_set = set()
        
        # 获取停用词文件路径
        current_dir = Path(__file__).parent
        stopwords_dir = current_dir / "stopwords"
        
        # 读取所有停用词文件
        stopwords_files = [
            stopwords_dir / "baidu_stopwords.txt",
            stopwords_dir / "cn_stopwords.txt"
        ]
        
        for file_path in stopwords_files:
            try:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 读取每行并去除空白字符
                        words = {line.strip() for line in f if line.strip()}
                        stopwords_set.update(words)
                        logger.info(f"从 {file_path} 加载了 {len(words)} 个中文停用词")
                else:
                    logger.warning(f"中文停用词文件 {file_path} 不存在")
            except Exception as e:
                logger.error(f"加载停用词文件 {file_path} 时出错: {e}")
        
        if not stopwords_set:
            logger.warning("未能加载任何中文停用词文件")
            
        return stopwords_set
        
    def clean_markdown(self, text: str) -> str:
        """清理Markdown标记
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        # 移除代码块
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # 移除行内代码
        text = re.sub(r'`[^`]*`', '', text)
        
        # 移除链接
        text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
        
        # 移除图片
        text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', '', text)
        
        # 移除标题标记
        text = re.sub(r'#{1,6}\s+', '', text)
        
        # 移除强调标记
        text = re.sub(r'[*_]{1,2}([^*_]*)[*_]{1,2}', r'\1', text)
        
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """移除停用词
        
        Args:
            text: 输入文本
            
        Returns:
            移除停用词后的文本
        """
        # 分词处理
        words = []
        
        # 使用jieba进行中文分词
        segments = jieba.cut(text)
        
        # 过滤停用词
        for word in segments:
            # 跳过标点符号,但保留 @ / .
            if re.match(r'[^\w\s@/.]', word):
                continue
                
            # 跳过中文停用词
            if word in self.zh_stopwords:
                continue
                
            # 跳过英文停用词
            if word.lower() in self.en_stopwords:
                continue
                
            words.append(word)
        
        return ''.join(words)
    
    def preprocess(self, text: str) -> str:
        """执行完整的文本预处理
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的文本
        """
        # 清理Markdown标记
        text = self.clean_markdown(text)
        
        # 移除停用词
        text = self.remove_stopwords(text)
        
        return text 