"""
# 并行计算工具模块 (Parallel Computing Utilities Module)
#
# 本文件提供了多线程和多进程并行计算的工具。主要内容包括：
#
# 1. ParallelExecutor：并行执行框架
#    - 支持线程和进程并行
#    - 处理字典和列表数据
#    - 显示进度条
#
# 2. 辅助功能：
#    - retry_on_error：错误重试装饰器
#    - parallel_cache：支持并行的结果缓存
#    - ChunkProcessor：大数据集分块处理
#
# 与其他组件的关系：
# - 被alphagen/models中的因子池使用，加速因子计算
# - 被scripts中的训练脚本使用，并行处理数据
# - 为alphagen/data中的数据处理提供并行支持
"""
import os
import pickle
import logging
import hashlib
from typing import Dict, List, TypeVar, Callable, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, partial
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

class ParallelExecutor:
    """Unified parallel execution framework with automatic worker type selection"""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = True,
        show_progress: bool = True
    ):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.show_progress = show_progress
        
    def get_executor(self):
        """Get appropriate executor based on task type"""
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        return executor_class(max_workers=self.max_workers)
    
    def map(
        self,
        func: Callable[[T], R],
        items: List[T],
        show_progress: Optional[bool] = None,
        desc: Optional[str] = None
    ) -> List[R]:
        """Execute function on items in parallel"""
        show_progress = self.show_progress if show_progress is None else show_progress
        
        with self.get_executor() as executor:
            futures = [executor.submit(func, item) for item in items]
            
            if show_progress:
                futures = tqdm(futures, desc=desc or "Processing")
            
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error in parallel execution: {str(e)}")
                    results.append(None)
            
            return results
    
    def map_dict(
        self,
        func: Callable[[str, T], R],
        items: Dict[str, T],
        show_progress: Optional[bool] = None,
        desc: Optional[str] = None
    ) -> Dict[str, R]:
        """Execute function on dictionary items in parallel"""
        show_progress = self.show_progress if show_progress is None else show_progress
        
        with self.get_executor() as executor:
            futures = {
                executor.submit(func, key, value): key 
                for key, value in items.items()
            }
            
            if show_progress:
                futures_iter = tqdm(futures.items(), desc=desc or "Processing")
            else:
                futures_iter = futures.items()
            
            results = {}
            for future, key in futures_iter:
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(f"Error processing {key}: {str(e)}")
                    results[key] = None
            
            return results

def retry_on_error(max_retries: int = 3, retry_delay: float = 1.0):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed, retrying: {str(e)}")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

def parallel_cache(cache_dir: str = ".cache"):
    """Decorator for caching function results with parallel support"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            key_parts = [
                func.__name__,
                str(args),
                str(sorted(kwargs.items()))
            ]
            cache_key = hashlib.md5(str(key_parts).encode()).hexdigest()
            
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # Try to load from cache
            try:
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed: {str(e)}")
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                logger.warning(f"Cache save failed: {str(e)}")
            
            return result
        return wrapper
    return decorator

class ChunkProcessor:
    """Process large datasets in chunks with parallel execution"""
    
    def __init__(
        self,
        chunk_size: int,
        n_workers: Optional[int] = None,
        use_processes: bool = True
    ):
        self.chunk_size = chunk_size
        self.executor = ParallelExecutor(
            max_workers=n_workers,
            use_processes=use_processes
        )
    
    def process(
        self,
        data: Union[List[T], Dict[str, T]],
        process_func: Callable[[T], R],
        show_progress: bool = True,
        desc: Optional[str] = None
    ) -> Union[List[R], Dict[str, R]]:
        """Process data in chunks"""
        if isinstance(data, dict):
            return self._process_dict(data, process_func, show_progress, desc)
        return self._process_list(data, process_func, show_progress, desc)
    
    def _process_list(
        self,
        data: List[T],
        process_func: Callable[[T], R],
        show_progress: bool,
        desc: Optional[str]
    ) -> List[R]:
        chunks = [
            data[i:i + self.chunk_size]
            for i in range(0, len(data), self.chunk_size)
        ]
        
        results = []
        for chunk in tqdm(chunks, desc=desc or "Processing chunks") if show_progress else chunks:
            chunk_results = self.executor.map(
                process_func,
                chunk,
                show_progress=False
            )
            results.extend(chunk_results)
        
        return results
    
    def _process_dict(
        self,
        data: Dict[str, T],
        process_func: Callable[[T], R],
        show_progress: bool,
        desc: Optional[str]
    ) -> Dict[str, R]:
        items = list(data.items())
        chunks = [
            dict(items[i:i + self.chunk_size])
            for i in range(0, len(items), self.chunk_size)
        ]
        
        results = {}
        for chunk in tqdm(chunks, desc=desc or "Processing chunks") if show_progress else chunks:
            chunk_results = self.executor.map_dict(
                lambda k, v: (k, process_func(v)),
                chunk,
                show_progress=False
            )
            results.update(chunk_results)
        
        return results 