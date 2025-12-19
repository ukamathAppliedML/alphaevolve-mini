#!/usr/bin/env python3
"""
Quick diagnostic to test if the LLM is generating usable code.
Run this first to verify your setup works.
"""

import asyncio
import sys
import os
import logging

# Suppress verbose httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.local_llm import LocalLLMEnsemble, check_ollama_status


SIMPLE_PROMPT = """Write a Python function that packs 3 circles into a unit square [0,1]×[0,1].

RULES:
- Only use standard library: math, random (NO scipy, numpy, or other external packages)
- Each circle is a tuple: (x, y, radius)
- Circles must not overlap
- Circles must fit entirely within the square
- Maximize sum of radii

Write a COMPLETE, SHORT function:

```python
def pack_circles():
    import math
    # Place 3 non-overlapping circles
    # Return list of (x, y, radius) tuples
    circles = []
    # Your simple implementation here
    return circles
```"""


async def test_generation(model: str = "qwen2.5-coder:7b"):
    print(f"Testing model: {model}\n")
    
    # Check status
    status = await check_ollama_status()
    print(f"Ollama running: {status['running']}")
    print(f"Models available: {status['models']}\n")
    
    if not status['running']:
        print("Start Ollama with: ollama serve")
        return
    
    # Test generation
    llm = LocalLLMEnsemble(provider="ollama", model_name=model)
    
    print("Sending prompt...")
    print("-" * 50)
    
    response = await llm.generate(SIMPLE_PROMPT, temperature=0.7, max_tokens=1000)
    
    print("Response:")
    print("-" * 50)
    print(response.content)
    print("-" * 50)
    print(f"\nTokens: {response.usage}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    
    # Try to extract and run
    import re
    match = re.search(r'def pack_circles\(\):(.*?)(?=\ndef |\Z)', response.content, re.DOTALL)
    
    if match or 'def pack_circles' in response.content:
        print("\n✓ Found function definition")
        
        # Try to execute
        try:
            # Extract code
            if '```python' in response.content:
                code = re.search(r'```python\s*\n(.*?)```', response.content, re.DOTALL)
                code = code.group(1) if code else response.content
            else:
                code = response.content
            
            if 'def pack_circles' in code:
                exec_globals = {}
                exec(code, exec_globals)
                
                if 'pack_circles' in exec_globals:
                    circles = exec_globals['pack_circles']()
                    print(f"✓ Function executed successfully")
                    print(f"  Circles: {circles}")
                    print(f"  Sum of radii: {sum(r for x,y,r in circles):.4f}")
                else:
                    print("✗ Function not found after exec")
            else:
                print("✗ No function definition in extracted code")
                
        except Exception as e:
            print(f"✗ Execution error: {e}")
    else:
        print("\n✗ No function definition found in response")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder:7b"
    asyncio.run(test_generation(model))
