from typing import Dict, Optional
from datetime import datetime, timedelta

def parse_metadata_filters(filter_str: str) -> Optional[Dict]:
    """解析元数据过滤字符串
    
    Args:
        filter_str: 过滤条件字符串，格式如：type:md,created_after:7,title:示例
        
    Returns:
        解析后的过滤条件字典
    """
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