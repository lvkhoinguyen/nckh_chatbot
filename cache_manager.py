from functools import lru_cache
import hashlib

class CacheManager:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_cache_key(self, data):
        """Generate unique cache key"""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        return hashlib.md5(str(data).encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def get_cached_response(self, question, context_hash):
        """Get cached response using LRU cache"""
        cache_key = self.get_cache_key(f"{question}_{context_hash}")
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None