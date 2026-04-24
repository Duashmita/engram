import os

GEMINI_CHAT_MODEL = 'gemini-2.0-flash'
GEMINI_EMBED_MODEL = 'text-embedding-004'
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

SESSION_WINDOW = 7        # max turns before eviction
EVICT_BATCH = 5           # turns evicted at once (triggers one summary)
RETRIEVAL_THRESHOLD = 15.0
KEY_MEMORY_PERCENTILE = 0.75  # top 25% by score → key memories
TOP_K_RETRIEVAL = 5
DECAY_RATE = 0.1
STORAGE_THRESHOLD = 0.2
THREAT_MAX_TOKENS = 200
