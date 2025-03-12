from pathlib import Path
from datetime import datetime, timedelta
from src.processors.markdown import MarkdownProcessor
from src.retrievers.hybrid_retriever import HybridRetriever
from src.rankers.reranker import Reranker
from src.config.settings import (
    DATA_DIR,
    INITIAL_RETRIEVAL_SIZE,
    FINAL_RETRIEVAL_SIZE
)

def parse_metadata_filters(filter_str: str) -> dict:
    """解析元数据过滤字符串"""
    if not filter_str:
        return None

    filters = {}
    parts = filter_str.split(',')
    for part in parts:
        if ':' not in part:
            continue
        key, value = part.strip().split(':', 1)
        
        # 处理时间范围
        if key in ['created_after', 'modified_after']:
            field = 'created_at' if 'created' in key else 'modified_at'
            days = int(value)
            timestamp = datetime.now() - timedelta(days=days)
            filters[field] = {'gte': timestamp.timestamp()}
        # 处理文件类型
        elif key == 'type':
            filters['file_type'] = value
        # 处理标题搜索
        elif key == 'title':
            filters['title'] = {'contains': value}
        else:
            filters[key] = value
            
    return filters

def main():
    # 初始化组件
    doc_processor = MarkdownProcessor()
    retriever = HybridRetriever()
    reranker = Reranker()

    # 处理文档
    print("开始处理文档...")
    documents = doc_processor.process_directory(DATA_DIR)
    print(f"处理完成，共获取 {len(documents)} 个文档片段")

    # 存储向量
    print("开始向量化并存储文档...")
    retriever.add_documents(documents)
    print("文档存储完成")

    # 搜索循环
    print("\n支持的过滤条件：")
    print("- type:md - 按文件类型过滤")
    print("- created_after:7 - 搜索7天内创建的文档")
    print("- modified_after:7 - 搜索7天内修改的文档")
    print("- title:关键词 - 搜索标题包含关键词的文档")
    print("多个条件用逗号分隔，例如：type:md,created_after:7")
    print("\n搜索选项：")
    print("- hybrid=true/false - 是否使用混合检索（默认：true）")
    print("- weight=0.7 - 向量搜索权重（默认：0.7）")
    print("- rerank=true/false - 是否使用重排序（默认：true）")

    while True:
        query = input("\n请输入搜索查询（输入'q'退出）: ")
        if query.lower() == 'q':
            break

        # 解析搜索选项
        options = {'use_hybrid': True}
        use_rerank = True
        if ' --' in query:
            query, options_str = query.split(' --', 1)
            for opt in options_str.split(' --'):
                if '=' in opt:
                    k, v = opt.split('=', 1)
                    if k == 'hybrid':
                        options['use_hybrid'] = v.lower() == 'true'
                    elif k == 'weight':
                        options['vector_weight'] = float(v)
                    elif k == 'rerank':
                        use_rerank = v.lower() == 'true'

        # 解析过滤条件
        filters = None
        if ' filter=' in query:
            query, filter_str = query.split(' filter=', 1)
            filters = parse_metadata_filters(filter_str)

        # 执行初始检索
        initial_results = retriever.search(
            query.strip(),
            limit=INITIAL_RETRIEVAL_SIZE if use_rerank else FINAL_RETRIEVAL_SIZE,
            metadata_filters=filters,
            **options
        )

        # 执行重排序（如果启用）
        results = reranker.rerank(query.strip(), initial_results) if use_rerank else initial_results

        # 显示结果
        print("\n搜索结果：")
        for i, hit in enumerate(results, 1):
            print(f"\n{i}. 相关度分数: {hit['score']:.4f}")
            if 'original_score' in hit:
                print(f"   原始分数: {hit['original_score']:.4f}")
            print(f"文件: {hit['file_name']}")
            if hit['title']:
                print(f"标题: {hit['title']}")
            print(f"创建时间: {hit['created_at']}")
            print(f"修改时间: {hit['modified_at']}")
            print(f"内容: {hit['content'][:200]}...")

if __name__ == "__main__":
    main() 