"""Run with: GEMINI_API_KEY=<key> python3 list_models.py"""
import os, warnings
warnings.filterwarnings("ignore")
from google import genai

key = os.environ.get("GEMINI_API_KEY", "")
if not key:
    print("Set GEMINI_API_KEY first"); exit(1)

client = genai.Client(api_key=key)
all_models = list(client.models.list())

print("\nAll models with 'embed' in name or supporting embedContent:\n")
for m in all_models:
    methods = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", [])
    if "embed" in m.name.lower() or "embedContent" in methods:
        print(f"  {m.name}")
        if methods:
            print(f"    methods: {methods}")

print("\nAll models (for reference):\n")
for m in all_models:
    print(f"  {m.name}")
